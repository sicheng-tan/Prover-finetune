#!/bin/bash
sketch_dir=/rjs/ghyx/llm/codes/FLRQ
data_base_dir=/rjs/ghyx/data
lm_eval_dir=/home/ghyx/llm/codes/lm-evaluation-harness
ppl_eval_dir=.

declare -a qbit_s=(3)
declare -a models=("Qwen2-5-coder-7B-9300")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" "opt-13b")

declare -a ratios=(0.2)

divider="------------------------------------------------------------------------"

for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle"
    echo "$divider"
    for ratio in "${ratios[@]}"; do
      echo ">>>>>>>> ratio = $ratio, bit = $bit"
      python $sketch_dir/run_quantize.py \
          --model_path "$data_base_dir/$testmodle" \
          --output_path "$data_base_dir/${testmodle}-3bit" \
          --qbit "$bit" \
          --groupsize 128 \
          --lora_ratio $ratio \
          --fix_rank 0

      echo ">>>>>>>>Starting test ppl: $testmodle, ratio:$ratio"
      
      CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/qwen.py "$data_base_dir/${testmodle}-3bit" "wikitext2"

      echo ">>>>>>>>Starting zero shot task: $testmodle, ratio:$ratio"
      lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks arc_challenge,arc_easy,winogrande,arc_challenge,piqa,boolq,openbookqa

      echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
      rm -rf $data_base_dir/${testmodle}-calib

    done
  done
done

