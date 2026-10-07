import argparse
import json
import logging
import os
import re

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--test_filename", type=str, default="/")
    parser.add_argument("--output_dir", type=str, default="/")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)

    return parser.parse_args()


def check_len(src_line, tokenizer, args, index):
    input_tokens = tokenizer.tokenize(src_line)

    if len(input_tokens) < args.max_source_length - 128:
        return src_line

    while True:
        input_tokens = tokenizer.tokenize(src_line)

        if len(input_tokens) > args.max_source_length - 128:
            bug_loc = input_tokens.index("start")

            if bug_loc + 256 < args.max_source_length - 128:
                src_line = src_line[: len(src_line) - 2]
            elif bug_loc - 256 > args.max_source_length - 128:
                src_line = src_line[1:]
            else:
                src_line = src_line[1:]
        else:
            break

    return src_line


def load_model_and_tokenizer(args):
    logger.info("-----------------------------------")
    logger.info("    Load Tokenizer and Model...    ")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    int4_config = BitsAndBytesConfig(load_in_4bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=int4_config,
        device_map="auto",
    )

    base_model = prepare_model_for_kbit_training(base_model)

    # Load the trained LoRA adapter onto the quantized backbone model.
    model = PeftModel.from_pretrained(
        base_model,
        args.peft_model_path,
    )

    # Merge the LoRA adapter into the backbone for inference.
    model = model.merge_and_unload()

    return model, tokenizer


def model_inference():
    model, tokenizer = load_model_and_tokenizer(args)

    logger.info("-----------------------------------")
    logger.info("            Start Test...          ")

    assert len(args.test_filename.split(",")) == 2
    src_filename = args.test_filename.split(",")[0]
    trg_filename = args.test_filename.split(",")[1]

    with open(trg_filename, "r", encoding="utf-8") as data_file:
        data_lines = data_file.readlines()
    data_size = len(data_lines)

    pre_list = []
    tgt_list = []
    good = 0
    index = 0

    with open(src_filename, "r", encoding="utf-8") as f1, open(
        trg_filename, "r", encoding="utf-8"
    ) as f2:
        logger.info("Data Size: %s", str(data_size))

        for line1, line2 in zip(f1, f2):
            index += 1

            src_dic = json.loads(line1)
            tgt_dic = json.loads(line2)

            src_line = src_dic["INPUT"]
            tgt_line = tgt_dic["INPUT"]

            tgt_list.append(str([tgt_line]) + "\n")

            src_line = check_len(src_line, tokenizer, args, index)

            input_text = "<bug_method>\n" + src_line + "\n<fix_code>\n"
            output_text = tgt_line

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
            pre_out = []

            if input_ids.size(1) >= args.max_source_length:
                pre_out.append("Input Too Long")
                logger.info(
                    " %d/%d, Input_len= %s, Pre_Output = %s",
                    index,
                    data_size,
                    str(input_ids.size(1)),
                    str(["Input Too Long"]),
                )
                pre_list.append(str(pre_out) + "\n")
                continue

            try:
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.beam_size,
                    num_return_sequences=args.output_size,
                    early_stopping=True,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                )

                for generated_id in generated_ids:
                    generated_text = tokenizer.decode(
                        generated_id,
                        skip_special_tokens=False,
                    )

                    if "\n<fix_code>\n" in generated_text:
                        text = generated_text.split("\n<fix_code>\n", 1)[1]
                    else:
                        text = generated_text

                    if tokenizer.eos_token:
                        text = text.replace(tokenizer.eos_token, "")

                    text = (
                        text.replace("<｜end▁of▁sentence｜>", "")
                        .replace("<｜begin▁of▁sentence｜>", "")
                        .replace("<eos>", "")
                    )
                    pre_out.append(text)

            except Exception as exc:
                pre_out.append("// Error\n")
                logger.info("%s", str(exc))

            em_result = "bad"

            for pre_one in pre_out:
                pre_txt = (
                    pre_one.replace("// fix_start\n", "")
                    .replace("// fix_end\n", "")
                    .replace("\n", "")
                )
                pre_txt = re.sub(r"\s+", "", pre_txt)

                output_txt = (
                    output_text.replace("// fix_start\n", "")
                    .replace("// fix_end\n", "")
                    .replace("\n", "")
                )
                output_txt = re.sub(r"\s+", "", output_txt)

                if pre_txt.strip() == output_txt.strip():
                    good += 1
                    em_result = "good"
                    break

            logger.info(
                " %d/%d, Input_len= %s, EM_result=%s, Beam_size=%s, Pre_patch=%s",
                index,
                data_size,
                str(input_ids.size(1)),
                em_result,
                str(args.beam_size),
                str(pre_out),
            )
            pre_list.append(str(pre_out) + "\n")

    logger.info(" Good = %d", good)
    logger.info(" All = %d", index)
    logger.info(" Repair Accuracy = %s", str(good / index))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        os.path.join(args.output_dir, "pre_file.txt"),
        "w",
        encoding="utf-8",
    ) as pre_file, open(
        os.path.join(args.output_dir, "tgt_file.txt"),
        "w",
        encoding="utf-8",
    ) as tgt_file:
        for pre, tgt in zip(pre_list, tgt_list):
            pre_file.write(pre)
            tgt_file.write(tgt)


if __name__ == "__main__":
    args = get_args()
    logger.info(args)

    model_inference()
