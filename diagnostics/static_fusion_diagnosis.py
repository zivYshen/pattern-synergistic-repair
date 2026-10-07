import argparse
import codecs
import csv
import json
import logging
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--ELR", type=str, default="/")
    parser.add_argument("--CMR", type=str, default="/")
    parser.add_argument("--SLR", type=str, default="/")
    parser.add_argument("--DCR", type=str, default="/")

    parser.add_argument("--adapters", type=str, default="ELR,CMR,SLR,DCR")
    parser.add_argument("--weights", type=str, default="1.0,1.0,1.0,1.0")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--density_list", type=str, default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--combination_type", type=str, default="ties",
                        choices=["ties", "dare_linear", "dare_ties"])

    parser.add_argument("--analysis_mode", type=str, default="none",
                        choices=[
                            "none",
                            "general_conflict",
                            "ties_retention",
                            "dare_cancellation",
                            "compare_em_logs",
                        ])

    parser.add_argument("--intervention", type=str, default="none",
                        choices=[
                            "none",
                            "high_cancellation_mask",
                            "high_cancellation_random_mask",
                            "high_cancellation_downweight",
                        ])

    parser.add_argument("--conflict_scale", type=float, default=1.0)
    parser.add_argument("--cancellation_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_list", type=str, default="42,123,2026")

    parser.add_argument("--single_adapter", type=str, default="",
                        help="Run inference using a single adapter, e.g., ELR.")

    parser.add_argument("--test_filename", type=str, default="TRANSFER")
    parser.add_argument("--output_dir", type=str, default="TRANSFER")
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--output_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=1)

    parser.add_argument("--log_paths", type=str, default="")
    parser.add_argument("--method_names", type=str, default="")
    parser.add_argument("--reference_name", type=str, default="")

    return parser.parse_args()


def parse_list_arg(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list_arg(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list_arg(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_target_lora_module(module_name, target_modules):
    return any(module_name.endswith(t) for t in target_modules)


def iter_lora_parameter_groups(model, adapters, target_modules):
    for module_name, module in model.named_modules():
        if not is_target_lora_module(module_name, target_modules):
            continue

        for side in ["lora_A", "lora_B"]:
            if not hasattr(module, side):
                continue

            lora_dict = getattr(module, side)
            if not all(adapter in lora_dict for adapter in adapters):
                continue

            tensors = [lora_dict[adapter].weight for adapter in adapters]
            if not all(t.shape == tensors[0].shape for t in tensors):
                continue

            yield module_name, side, tensors


def trim_topk_per_expert(stacked, density):
    """
    TIES-style trimming.
    density=0.4 means keeping top 40% magnitude values for each expert.
    """
    if density <= 0.0 or density > 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    trimmed = torch.zeros_like(stacked)
    thresholds = []

    for i in range(stacked.size(0)):
        flat = stacked[i].reshape(-1)
        abs_flat = flat.abs()
        numel = abs_flat.numel()

        k = int(round(numel * density))
        k = max(1, min(k, numel))

        top_values, top_indices = torch.topk(abs_flat, k=k, largest=True, sorted=False)

        trimmed_flat = torch.zeros_like(flat)
        trimmed_flat[top_indices] = flat[top_indices]
        trimmed[i] = trimmed_flat.reshape_as(stacked[i])

        thresholds.append(float(top_values.min().item()))

    return trimmed, thresholds


def build_weighted_stack(tensors, weights):
    stacked = torch.stack([t.detach().float().cpu() for t in tensors], dim=0)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype).view(
        -1, *([1] * (stacked.dim() - 1))
    )
    return stacked * weight_tensor


def build_general_sign_conflict_masks(tensors, weights, density):
    """
    General sign disagreement:
    after magnitude trimming, a coordinate has sign conflict if positive
    and negative expert updates both exist.
    """
    weighted = build_weighted_stack(tensors, weights)
    trimmed, thresholds = trim_topk_per_expert(weighted, density)

    signs = torch.sign(trimmed).to(torch.int8)
    active_value_mask = signs != 0
    active_coordinate_mask = active_value_mask.any(dim=0)

    has_pos = (signs == 1).any(dim=0)
    has_neg = (signs == -1).any(dim=0)

    coordinate_conflict_mask = active_coordinate_mask & has_pos & has_neg
    all_agree_coordinate_mask = active_coordinate_mask & (
        (has_pos & ~has_neg) | (~has_pos & has_neg)
    )

    elected_sign = torch.sign(trimmed.sum(dim=0)).to(torch.int8)
    elected_nonzero = elected_sign != 0

    value_conflict_mask = active_value_mask & elected_nonzero.unsqueeze(0) & (
        signs != elected_sign.unsqueeze(0)
    )
    value_aligned_mask = active_value_mask & elected_nonzero.unsqueeze(0) & (
        signs == elected_sign.unsqueeze(0)
    )

    stats = {
        "total_coordinates": int(active_coordinate_mask.numel()),
        "active_coordinates": int(active_coordinate_mask.sum().item()),
        "coordinate_conflict": int(coordinate_conflict_mask.sum().item()),
        "all_agree_coordinates": int(all_agree_coordinate_mask.sum().item()),
        "active_values": int(active_value_mask.sum().item()),
        "conflicting_values": int(value_conflict_mask.sum().item()),
        "aligned_values": int(value_aligned_mask.sum().item()),
        "avg_trim_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
    }

    return value_conflict_mask, value_aligned_mask, coordinate_conflict_mask, active_coordinate_mask, stats


def run_general_conflict_analysis(model, adapters, weights, target_modules, density_list, output_dir):
    rows = []

    for density in density_list:
        agg = {
            "total_coordinates": 0,
            "active_coordinates": 0,
            "coordinate_conflict": 0,
            "all_agree_coordinates": 0,
            "active_values": 0,
            "conflicting_values": 0,
            "aligned_values": 0,
        }
        thresholds = []

        for module_name, side, tensors in iter_lora_parameter_groups(model, adapters, target_modules):
            _, _, _, _, stats = build_general_sign_conflict_masks(
                tensors=tensors,
                weights=weights,
                density=density,
            )

            for k in agg:
                agg[k] += stats[k]
            thresholds.append(stats["avg_trim_threshold"])

        active_coordinates = max(agg["active_coordinates"], 1)
        active_values = max(agg["active_values"], 1)

        rows.append({
            "density": density,
            "avg_trim_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
            "active_coordinate_coverage": agg["active_coordinates"] / max(agg["total_coordinates"], 1),
            "coordinate_sign_conflict_ratio": agg["coordinate_conflict"] / active_coordinates,
            "all_agree_coordinate_ratio": agg["all_agree_coordinates"] / active_coordinates,
            "value_conflict_ratio": agg["conflicting_values"] / active_values,
            "aligned_value_ratio": agg["aligned_values"] / active_values,
            "active_coordinates": agg["active_coordinates"],
            "coordinate_conflict": agg["coordinate_conflict"],
            "active_values": agg["active_values"],
            "conflicting_values": agg["conflicting_values"],
            "aligned_values": agg["aligned_values"],
            "total_coordinates": agg["total_coordinates"],
        })

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "general_sign_disagreement_sensitivity.csv")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved general sign disagreement analysis to %s", out_file)

    for row in rows:
        logger.info(
            "density=%.2f, coordinate_conflict=%.4f, value_conflict=%.4f",
            row["density"],
            row["coordinate_sign_conflict_ratio"],
            row["value_conflict_ratio"],
        )


def run_ties_retention_analysis(model, adapters, weights, target_modules, density, output_dir):
    """
    Analyze how many active values from each expert are retained or discarded
    by TIES-style global sign election.
    """
    per_expert = {
        adapter: {
            "active_values": 0,
            "retained_values": 0,
            "discarded_values": 0,
        }
        for adapter in adapters
    }

    for module_name, side, tensors in iter_lora_parameter_groups(model, adapters, target_modules):
        value_conflict_mask, value_aligned_mask, _, _, _ = build_general_sign_conflict_masks(
            tensors=tensors,
            weights=weights,
            density=density,
        )

        for expert_idx, adapter in enumerate(adapters):
            active = int((value_conflict_mask[expert_idx] | value_aligned_mask[expert_idx]).sum().item())
            retained = int(value_aligned_mask[expert_idx].sum().item())
            discarded = int(value_conflict_mask[expert_idx].sum().item())

            per_expert[adapter]["active_values"] += active
            per_expert[adapter]["retained_values"] += retained
            per_expert[adapter]["discarded_values"] += discarded

    rows = []
    for adapter, stats in per_expert.items():
        active = max(stats["active_values"], 1)
        rows.append({
            "expert": adapter,
            "active_values": stats["active_values"],
            "retained_values": stats["retained_values"],
            "discarded_values": stats["discarded_values"],
            "retention_ratio": stats["retained_values"] / active,
            "discard_ratio": stats["discarded_values"] / active,
        })

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "ties_expert_retention.csv")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved TIES expert retention analysis to %s", out_file)
    for row in rows:
        logger.info(
            "%s: active=%d, retained=%d, discarded=%d, retention=%.4f",
            row["expert"],
            row["active_values"],
            row["retained_values"],
            row["discarded_values"],
            row["retention_ratio"],
        )


def cancellation_stats_from_weighted(weighted, cancellation_threshold=0.5, eps=1e-12):
    """
    C_p = 1 - |sum_i x_i| / (sum_i |x_i| + eps)
    Higher means stronger cancellation.
    """
    sum_signed = weighted.sum(dim=0)
    sum_abs = weighted.abs().sum(dim=0)

    active = sum_abs > eps
    cancellation = torch.zeros_like(sum_abs)
    cancellation[active] = 1.0 - (sum_signed.abs()[active] / (sum_abs[active] + eps))

    high = active & (cancellation >= cancellation_threshold)

    active_count = int(active.sum().item())
    high_count = int(high.sum().item())
    mean_cancellation = float(cancellation[active].mean().item()) if active_count > 0 else 0.0

    return {
        "active_coordinates": active_count,
        "high_cancellation_coordinates": high_count,
        "high_cancellation_ratio": high_count / max(active_count, 1),
        "mean_cancellation": mean_cancellation,
    }


def apply_dare_drop_to_weighted(weighted, density, seed):
    """
    Approximate DARE's drop-and-rescale on delta parameters.
    density is the keep ratio. For example, density=0.8 keeps 80% and rescales by 1/0.8.
    """
    if density <= 0.0 or density > 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    keep_mask = torch.rand(weighted.shape, generator=rng) < density
    return weighted * keep_mask.to(weighted.dtype) / density


def run_dare_cancellation_analysis(
    model,
    adapters,
    weights,
    target_modules,
    density,
    seed_list,
    cancellation_threshold,
    output_dir,
):
    rows = []

    aggregate_before = {
        "active_coordinates": 0,
        "high_cancellation_coordinates": 0,
        "mean_cancellation_weighted_sum": 0.0,
    }

    aggregate_after = {
        seed: {
            "active_coordinates": 0,
            "high_cancellation_coordinates": 0,
            "mean_cancellation_weighted_sum": 0.0,
        }
        for seed in seed_list
    }

    for module_name, side, tensors in iter_lora_parameter_groups(model, adapters, target_modules):
        weighted = build_weighted_stack(tensors, weights)

        before_stats = cancellation_stats_from_weighted(weighted, cancellation_threshold)
        aggregate_before["active_coordinates"] += before_stats["active_coordinates"]
        aggregate_before["high_cancellation_coordinates"] += before_stats["high_cancellation_coordinates"]
        aggregate_before["mean_cancellation_weighted_sum"] += (
            before_stats["mean_cancellation"] * before_stats["active_coordinates"]
        )

        for seed in seed_list:
            weighted_after = apply_dare_drop_to_weighted(weighted, density=density, seed=seed)
            after_stats = cancellation_stats_from_weighted(weighted_after, cancellation_threshold)
            aggregate_after[seed]["active_coordinates"] += after_stats["active_coordinates"]
            aggregate_after[seed]["high_cancellation_coordinates"] += after_stats["high_cancellation_coordinates"]
            aggregate_after[seed]["mean_cancellation_weighted_sum"] += (
                after_stats["mean_cancellation"] * after_stats["active_coordinates"]
            )

    active_before = max(aggregate_before["active_coordinates"], 1)
    rows.append({
        "setting": "before_dare_drop",
        "seed": "-",
        "density": density,
        "mean_cancellation": aggregate_before["mean_cancellation_weighted_sum"] / active_before,
        "high_cancellation_ratio": aggregate_before["high_cancellation_coordinates"] / active_before,
        "active_coordinates": aggregate_before["active_coordinates"],
        "high_cancellation_coordinates": aggregate_before["high_cancellation_coordinates"],
    })

    for seed in seed_list:
        active_after = max(aggregate_after[seed]["active_coordinates"], 1)
        rows.append({
            "setting": "after_dare_drop",
            "seed": seed,
            "density": density,
            "mean_cancellation": aggregate_after[seed]["mean_cancellation_weighted_sum"] / active_after,
            "high_cancellation_ratio": aggregate_after[seed]["high_cancellation_coordinates"] / active_after,
            "active_coordinates": aggregate_after[seed]["active_coordinates"],
            "high_cancellation_coordinates": aggregate_after[seed]["high_cancellation_coordinates"],
        })

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "dare_cancellation_analysis.csv")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved DARE cancellation analysis to %s", out_file)
    for row in rows:
        logger.info(
            "%s seed=%s: mean_cancel=%.4f, high_cancel=%.4f",
            row["setting"],
            row["seed"],
            row["mean_cancellation"],
            row["high_cancellation_ratio"],
        )


def build_high_cancellation_mask(tensors, weights, cancellation_threshold, eps=1e-12):
    weighted = build_weighted_stack(tensors, weights)

    sum_signed = weighted.sum(dim=0)
    sum_abs = weighted.abs().sum(dim=0)
    active = sum_abs > eps

    cancellation = torch.zeros_like(sum_abs)
    cancellation[active] = 1.0 - (sum_signed.abs()[active] / (sum_abs[active] + eps))

    high = active & (cancellation >= cancellation_threshold)

    return high, active, cancellation


def apply_high_cancellation_intervention(
    model,
    adapters,
    weights,
    target_modules,
    mode,
    cancellation_threshold,
    seed=42,
    conflict_scale=1.0,
):
    assert mode in [
        "high_cancellation_mask",
        "high_cancellation_random_mask",
        "high_cancellation_downweight",
    ]

    if mode == "high_cancellation_downweight":
        if conflict_scale < 0.0 or conflict_scale > 1.0:
            raise ValueError(f"conflict_scale must be in [0, 1], got {conflict_scale}")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    total_affected = 0

    with torch.no_grad():
        for module_name, side, tensors in iter_lora_parameter_groups(model, adapters, target_modules):
            high_mask_cpu, active_mask_cpu, cancellation = build_high_cancellation_mask(
                tensors=tensors,
                weights=weights,
                cancellation_threshold=cancellation_threshold,
            )

            if mode in ["high_cancellation_mask", "high_cancellation_downweight"]:
                coord_mask_cpu = high_mask_cpu
            else:
                candidate_cpu = active_mask_cpu & (~high_mask_cpu)
                candidate_idx = candidate_cpu.reshape(-1).nonzero(as_tuple=False).reshape(-1)

                num_to_mask = int(high_mask_cpu.sum().item())
                num_to_mask = min(num_to_mask, candidate_idx.numel())

                if num_to_mask == 0:
                    continue

                perm = torch.randperm(candidate_idx.numel(), generator=rng)[:num_to_mask]
                chosen = candidate_idx[perm]

                mask_flat = torch.zeros(candidate_cpu.numel(), dtype=torch.bool)
                mask_flat[chosen] = True
                coord_mask_cpu = mask_flat.reshape_as(candidate_cpu)

            affected_count = int(coord_mask_cpu.sum().item())
            total_affected += affected_count

            if affected_count == 0:
                continue

            for t in tensors:
                coord_mask = coord_mask_cpu.to(t.device)
                if mode == "high_cancellation_downweight":
                    t.data[coord_mask] *= conflict_scale
                else:
                    t.data[coord_mask] = 0

            logger.info(
                "[%s] %s.%s: affected_coordinates=%d, threshold=%.2f, scale=%.2f",
                mode,
                module_name,
                side,
                affected_count,
                cancellation_threshold,
                conflict_scale,
            )

    logger.info(
        "Total affected high-cancellation coordinates: %d, mode=%s, scale=%.2f",
        total_affected,
        mode,
        conflict_scale,
    )


def check_len(src_line, tokenizer, args, i):
    input_tokens = tokenizer.tokenize(src_line)

    if len(input_tokens) < args.max_source_length - 128:
        return src_line

    while True:
        input_tokens = tokenizer.tokenize(src_line)

        if len(input_tokens) > args.max_source_length - 128:
            try:
                bug_loc = input_tokens.index("start")
            except ValueError:
                src_line = src_line[1:]
                continue

            if bug_loc + 256 < args.max_source_length - 128:
                src_line = src_line[:len(src_line) - 2]
            else:
                src_line = src_line[1:]
        else:
            break

    return src_line


def run_inference(model, tokenizer, args):
    assert len(args.test_filename.split(",")) == 2
    src_filename = args.test_filename.split(",")[0]
    trg_filename = args.test_filename.split(",")[1]

    with open(trg_filename, "r", encoding="utf-8") as data_file:
        data_lines = data_file.readlines()
    data_size = len(data_lines)

    pre_list = []
    tgt_list = []
    good = 0
    i = 0

    with open(src_filename, "r", encoding="utf-8") as f1, open(trg_filename, "r", encoding="utf-8") as f2:
        logger.info("Data Size: %s", str(data_size))

        for line1, line2 in tqdm(zip(f1, f2), total=data_size, desc="Processing lines"):
            i += 1

            src_dic = json.loads(line1)
            tgt_dic = json.loads(line2)

            src_line = src_dic["INPUT"]
            tgt_line = tgt_dic["INPUT"]

            tgt_list.append(str([tgt_line]) + "\n")
            src_line = check_len(src_line, tokenizer, args, i)

            input_text = "<bug_method>\n" + src_line + "\n<fix_code>\n"
            output_text = tgt_line

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

            if input_ids.size(1) >= args.max_source_length:
                pre_list.append(str(["Input Too Long"]) + "\n")
                logger.info(
                    " %d/%d , Input_Len = %d, Pre_Output = %s",
                    i,
                    data_size,
                    input_ids.size(1),
                    str(["Input Too Long"]),
                )
                continue

            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

            try:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=args.max_target_length,
                        num_beams=args.beam_size,
                        num_return_sequences=args.output_size,
                        early_stopping=True,
                        pad_token_id=eos_id,
                        eos_token_id=eos_id,
                        use_cache=True,
                    )

                pre_out = []
                for generated_id in generated_ids:
                    generated_text = tokenizer.decode(generated_id, skip_special_tokens=False)
                    if "\n<fix_code>\n" in generated_text:
                        text = generated_text.split("\n<fix_code>\n", 1)[1]
                    else:
                        text = generated_text

                    text = (
                        text.split("</s>")[0]
                        .replace("<｜end▁of▁sentence｜>", "")
                        .replace("<｜begin▁of▁sentence｜>", "")
                    )
                    pre_out.append(text)

            except Exception as e:
                pre_out = ["// Error\n"]
                logger.info("%s", str(e))

            EM_result = "bad"

            for pre_one in pre_out:
                pre_txt = (
                    pre_one.replace("// fix_start\n", "")
                    .replace("// fix_start", "")
                    .replace("// fix_end\n", "")
                    .replace("// fix_end", "")
                    .replace("\n", "")
                )
                pre_txt = re.sub(r"\s+", "", pre_txt)

                output_txt = (
                    output_text.replace("// fix_start\n", "")
                    .replace("// fix_start", "")
                    .replace("// fix_end\n", "")
                    .replace("// fix_end", "")
                    .replace("\n", "")
                )
                output_txt = re.sub(r"\s+", "", output_txt)

                if pre_txt.strip() == output_txt.strip():
                    good += 1
                    EM_result = "good"
                    break

            logger.info(
                " %d/%d, Pre_EM = %s, Input_len = %d, Pre_Fix = %s",
                i,
                data_size,
                str(EM_result),
                input_ids.size(1),
                str(pre_out),
            )
            pre_list.append(str(pre_out) + "\n")

    logger.info(" Good = %d", good)
    logger.info(" All = %d", i)
    logger.info(" Repair Accuracy = %s", str(good / max(i, 1)))

    with codecs.open(os.path.join(args.output_dir, "pre_file.txt"), "w", encoding="utf-8") as pre_file, \
            codecs.open(os.path.join(args.output_dir, "tgt_file.txt"), "w", encoding="utf-8") as tgt_file:
        for pre, tgt in zip(pre_list, tgt_list):
            pre_file.write(pre)
            tgt_file.write(tgt)


def parse_last_run_em_vector(log_path):
    log_path = Path(log_path)
    pattern = re.compile(
        r"(?P<idx>\d+)\s*/\s*(?P<total>\d+)\s*,\s*Pre_EM\s*=\s*(?P<em>good|bad)",
        re.IGNORECASE,
    )

    runs = []
    current_run = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            idx = int(m.group("idx"))
            total = int(m.group("total"))
            em = m.group("em").lower()

            if idx == 1 and current_run:
                runs.append(current_run)
                current_run = []

            current_run.append((idx, total, 1 if em == "good" else 0))

    if current_run:
        runs.append(current_run)

    if not runs:
        raise ValueError(f"No Pre_EM records found in {log_path}")

    last_run = sorted(runs[-1], key=lambda x: x[0])
    return [x[2] for x in last_run]


def mcnemar_asymptotic_p_value(n10, n01):
    """
    Asymptotic McNemar test without continuity correction.
    """
    discordant = n10 + n01
    if discordant == 0:
        return 1.0

    chi2 = ((n10 - n01) ** 2) / discordant
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return p_value


def run_compare_em_logs(args):
    log_paths = parse_list_arg(args.log_paths)
    method_names = parse_list_arg(args.method_names)

    if len(log_paths) != len(method_names):
        raise ValueError("log_paths and method_names must have the same length.")

    vectors = {}
    for name, path in zip(method_names, log_paths):
        vectors[name] = parse_last_run_em_vector(path)
        logger.info("%s: %d / %d", name, sum(vectors[name]), len(vectors[name]))

    lengths = {len(v) for v in vectors.values()}
    if len(lengths) != 1:
        raise ValueError(f"EM vectors have different lengths: {lengths}")

    n = next(iter(lengths))
    os.makedirs(args.output_dir, exist_ok=True)

    per_defect_file = os.path.join(args.output_dir, "per_defect_em_vectors.csv")
    with open(per_defect_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + method_names)
        for i in range(n):
            writer.writerow([i + 1] + [vectors[name][i] for name in method_names])

    summary_rows = []
    for name in method_names:
        summary_rows.append({
            "method": name,
            "em_good": sum(vectors[name]),
            "total": n,
            "em_accuracy": sum(vectors[name]) / n,
        })

    summary_file = os.path.join(args.output_dir, "em_count_summary.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    ref_name = args.reference_name or method_names[0]
    if ref_name in vectors and len(method_names) > 1:
        ref_vec = vectors[ref_name]
        others = [name for name in method_names if name != ref_name]

        union_vec = []
        for i in range(n):
            union_vec.append(1 if any(vectors[name][i] == 1 for name in others) else 0)

        both = sum(1 for a, b in zip(ref_vec, union_vec) if a == 1 and b == 1)
        ref_only = sum(1 for a, b in zip(ref_vec, union_vec) if a == 1 and b == 0)
        union_only = sum(1 for a, b in zip(ref_vec, union_vec) if a == 0 and b == 1)
        neither = sum(1 for a, b in zip(ref_vec, union_vec) if a == 0 and b == 0)

        overlap_rows = [{
            "reference": ref_name,
            "expert_union": "+".join(others),
            "reference_good": sum(ref_vec),
            "expert_union_good": sum(union_vec),
            "both_good": both,
            "reference_only": ref_only,
            "expert_union_only": union_only,
            "neither_good": neither,
            "mcnemar_b_reference_only": ref_only,
            "mcnemar_c_union_only": union_only,
            "mcnemar_chi2": ((ref_only - union_only) ** 2) / (ref_only + union_only) if (ref_only + union_only) > 0 else 0.0,
            "mcnemar_p": mcnemar_asymptotic_p_value(ref_only, union_only),
        }]

        for expert in others:
            expert_vec = vectors[expert]
            expert_good_ref_bad = sum(
                1 for a, b in zip(ref_vec, expert_vec) if a == 0 and b == 1
            )
            ref_good_expert_bad = sum(
                1 for a, b in zip(ref_vec, expert_vec) if a == 1 and b == 0
            )

            overlap_rows.append({
                "reference": ref_name,
                "expert_union": expert,
                "reference_good": sum(ref_vec),
                "expert_union_good": sum(expert_vec),
                "both_good": sum(1 for a, b in zip(ref_vec, expert_vec) if a == 1 and b == 1),
                "reference_only": ref_good_expert_bad,
                "expert_union_only": expert_good_ref_bad,
                "neither_good": sum(1 for a, b in zip(ref_vec, expert_vec) if a == 0 and b == 0),
                "mcnemar_b_reference_only": ref_good_expert_bad,
                "mcnemar_c_union_only": expert_good_ref_bad,
                "mcnemar_chi2": ((ref_good_expert_bad - expert_good_ref_bad) ** 2) / (ref_good_expert_bad + expert_good_ref_bad) if (ref_good_expert_bad + expert_good_ref_bad) > 0 else 0.0,
                "mcnemar_p": mcnemar_asymptotic_p_value(ref_good_expert_bad, expert_good_ref_bad),
            })

        overlap_file = os.path.join(args.output_dir, "reference_vs_expert_union_overlap.csv")
        with open(overlap_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_rows)

        logger.info("Saved overlap analysis to %s", overlap_file)

    logger.info("Saved EM vectors to %s", per_defect_file)
    logger.info("Saved EM summary to %s", summary_file)


def load_model_and_adapters(args):
    logger.info("-----------------------------------")
    logger.info("Load Tokenizer and Model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    int8_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=int8_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(model, args.ELR, adapter_name="ELR")
    _ = model.load_adapter(args.CMR, adapter_name="CMR")
    _ = model.load_adapter(args.SLR, adapter_name="SLR")
    _ = model.load_adapter(args.DCR, adapter_name="DCR")

    return model, tokenizer


def main():
    global args
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "test.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        encoding="utf-8",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                  "%m/%d/%Y %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    global logger
    logger = logging.getLogger(__name__)

    logger.info(args)
    set_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    if args.analysis_mode == "compare_em_logs":
        run_compare_em_logs(args)
        return

    model, tokenizer = load_model_and_adapters(args)

    adapters = parse_list_arg(args.adapters)
    weights = parse_float_list_arg(args.weights)
    target_modules = parse_list_arg(args.target_modules)

    if len(adapters) != len(weights):
        raise ValueError("adapters and weights must have the same length.")

    logger.info("Adapters: %s", adapters)
    logger.info("Weights: %s", weights)
    logger.info("Combination type: %s", args.combination_type)
    logger.info("Density: %.4f", args.density)
    logger.info("Target modules: %s", target_modules)

    if args.analysis_mode == "general_conflict":
        run_general_conflict_analysis(
            model=model,
            adapters=adapters,
            weights=weights,
            target_modules=target_modules,
            density_list=parse_float_list_arg(args.density_list),
            output_dir=args.output_dir,
        )
        return

    if args.analysis_mode == "ties_retention":
        run_ties_retention_analysis(
            model=model,
            adapters=adapters,
            weights=weights,
            target_modules=target_modules,
            density=args.density,
            output_dir=args.output_dir,
        )
        return

    if args.analysis_mode == "dare_cancellation":
        run_dare_cancellation_analysis(
            model=model,
            adapters=adapters,
            weights=weights,
            target_modules=target_modules,
            density=args.density,
            seed_list=parse_int_list_arg(args.seed_list),
            cancellation_threshold=args.cancellation_threshold,
            output_dir=args.output_dir,
        )
        return

    if args.single_adapter:
        logger.info("Running single adapter inference: %s", args.single_adapter)
        model.set_adapter(args.single_adapter)

    else:
        if args.intervention != "none":
            apply_high_cancellation_intervention(
                model=model,
                adapters=adapters,
                weights=weights,
                target_modules=target_modules,
                mode=args.intervention,
                cancellation_threshold=args.cancellation_threshold,
                seed=args.seed,
                conflict_scale=args.conflict_scale,
            )

        adapter_name = "merge"
        model.add_weighted_adapter(
            adapters,
            weights,
            adapter_name,
            combination_type=args.combination_type,
            density=args.density,
        )
        model.set_adapter(adapter_name)

    logger.info("-----------------------------------")
    logger.info("Start Test")
    run_inference(model, tokenizer, args)


if __name__ == "__main__":
    main()
