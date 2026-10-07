@echo off
chcp 65001

call conda activate xlora

set SCRIPT=static_fusion_diagnosis.py
set MODEL=..\model\deepseek-coder-1.3B
set ELR=res_out\ELR\lora_Epoch_3
set CMR=res_out\CMR\lora_Epoch_4
set SLR=res_out\SLR\lora_Epoch_3
set DCR=res_out\DCR\lora_Epoch_2
set TEST=..\datasets\d4j\src-test.jsonl,..\datasets\d4j\tgt-test.jsonl

echo ========================================
echo 1. General sign disagreement for TIES setting
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --output_dir "diagnosis\general_conflict_ties" ^
  --analysis_mode general_conflict ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "1.0,1.0,1.0,1.0" ^
  --density_list "0.2,0.4,0.6,0.8,1.0" ^
  --target_modules "q_proj,v_proj"

echo ========================================
echo 2. General sign disagreement for DARE setting
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --output_dir "diagnosis\general_conflict_dare" ^
  --analysis_mode general_conflict ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density_list "0.2,0.4,0.6,0.8,1.0" ^
  --target_modules "q_proj,v_proj"

echo ========================================
echo 3. TIES expert retention / discarded values
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --output_dir "diagnosis\ties_retention" ^
  --analysis_mode ties_retention ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "1.0,1.0,1.0,1.0" ^
  --density 0.4 ^
  --target_modules "q_proj,v_proj"

echo ========================================
echo 4. Single expert inference: ELR
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\single_experts\ELR" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --single_adapter "ELR"

echo ========================================
echo 5. Single expert inference: CMR
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\single_experts\CMR" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --single_adapter "CMR"

echo ========================================
echo 6. Single expert inference: SLR
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\single_experts\SLR" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --single_adapter "SLR"

echo ========================================
echo 7. Single expert inference: DCR
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\single_experts\DCR" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --single_adapter "DCR"

echo ========================================
echo 8. Compare TIES original with single expert union
echo ========================================

python -u %SCRIPT% ^
  --analysis_mode compare_em_logs ^
  --output_dir "diagnosis\ties_single_expert_overlap" ^
  --log_paths "ties_merge\new4lora-d4j\ties_original\test.log,diagnosis\single_experts\ELR\test.log,diagnosis\single_experts\CMR\test.log,diagnosis\single_experts\SLR\test.log,diagnosis\single_experts\DCR\test.log" ^
  --method_names "TIES,ELR,CMR,SLR,DCR" ^
  --reference_name "TIES"

echo ========================================
echo 9. DARE cancellation analysis
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --output_dir "diagnosis\dare_cancellation" ^
  --analysis_mode dare_cancellation ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --seed_list "42,123,2026" ^
  --cancellation_threshold 0.5 ^
  --target_modules "q_proj,v_proj"

echo ========================================
echo 10. DARE original seed 123
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_seed\dare_original_seed123" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --seed 123


echo ========================================
echo 11. DARE original seed 2026
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_seed\dare_original_seed2026" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --seed 2026

echo ========================================
echo 12. DARE high-cancellation downweight scale 0.75
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_high_cancel\dare_hc_downweight_0.75" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --target_modules "q_proj,v_proj" ^
  --intervention high_cancellation_downweight ^
  --cancellation_threshold 0.5 ^
  --conflict_scale 0.75 ^
  --seed 42

echo ========================================
echo 13. DARE high-cancellation downweight scale 0.50
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_high_cancel\dare_hc_downweight_0.50" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --target_modules "q_proj,v_proj" ^
  --intervention high_cancellation_downweight ^
  --cancellation_threshold 0.5 ^
  --conflict_scale 0.50 ^
  --seed 42

echo ========================================
echo 14. DARE high-cancellation downweight scale 0.25
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_high_cancel\dare_hc_downweight_0.25" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --target_modules "q_proj,v_proj" ^
  --intervention high_cancellation_downweight ^
  --cancellation_threshold 0.5 ^
  --conflict_scale 0.25 ^
  --seed 42

echo ========================================
echo 15. DARE high-cancellation mask
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_high_cancel\dare_hc_mask" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --target_modules "q_proj,v_proj" ^
  --intervention high_cancellation_mask ^
  --cancellation_threshold 0.5 ^
  --seed 42

echo ========================================
echo 16. DARE high-cancellation random matched mask
echo ========================================

python -u %SCRIPT% ^
  --model_name_or_path "%MODEL%" ^
  --ELR "%ELR%" ^
  --CMR "%CMR%" ^
  --SLR "%SLR%" ^
  --DCR "%DCR%" ^
  --test_filename "%TEST%" ^
  --output_dir "diagnosis\dare_high_cancel\dare_hc_random_mask_seed42" ^
  --max_source_length 2048 ^
  --max_target_length 128 ^
  --beam_size 10 ^
  --output_size 10 ^
  --test_batch_size 1 ^
  --adapters "ELR,CMR,SLR,DCR" ^
  --weights "0.5,1.0,0.25,0.125" ^
  --density 0.8 ^
  --combination_type "dare_linear" ^
  --target_modules "q_proj,v_proj" ^
  --intervention high_cancellation_random_mask ^
  --cancellation_threshold 0.5 ^
  --seed 42

echo ========================================
echo 17. Compare DARE logs
echo ========================================

python -u %SCRIPT% ^
  --analysis_mode compare_em_logs ^
  --output_dir "diagnosis\dare_comparison_summary" ^
  --log_paths "dare_merge\new4lora-d4j\dare_original\test.log,diagnosis\dare_seed\dare_original_seed123\test.log,diagnosis\dare_seed\dare_original_seed2026\test.log,diagnosis\dare_high_cancel\dare_hc_downweight_0.75\test.log,diagnosis\dare_high_cancel\dare_hc_downweight_0.50\test.log,diagnosis\dare_high_cancel\dare_hc_downweight_0.25\test.log,diagnosis\dare_high_cancel\dare_hc_mask\test.log,diagnosis\dare_high_cancel\dare_hc_random_mask_seed42\test.log" ^
  --method_names "DARE_seed42,DARE_seed123,DARE_seed2026,HC_downweight_0.75,HC_downweight_0.50,HC_downweight_0.25,HC_mask,HC_random_mask" ^
  --reference_name "DARE_seed42"

echo ========================================
echo All static fusion diagnosis experiments finished.
echo ========================================

pause