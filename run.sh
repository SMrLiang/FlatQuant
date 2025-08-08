# W4A4KV4
GPU=4
export CUDA_VISIBLE_DEVICES=$GPU
# meta-llama/Llama-2-7b-hf
# /home/liangyiheng/xten/models/internlm/internlm2_1_8b/hf
# /home/liangyiheng/xten/models/qwen/qwen2-7b/hf
python ./main.py \
    --model /home/liangyiheng/xten/models/internlm/internlm2_1_8b/hf \
    --w_bits 8 --a_bits 8 \
    --k_bits 8 --k_asym --k_groupsize 128 \
    --v_bits 8 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --epochs 15 \
    --output_dir ./outputs --save_matrix 
    # --lm_eval --lm_eval_batch_size 16 \

    # --resume

# python ./main.py \
#     --model /home/liangyiheng/xten/models/qwen/qwen2-7b/hf \
#     --w_bits 4 --a_bits 4 \
#     --k_bits 4 --k_asym --k_groupsize 128 \
#     --v_bits 4 --v_asym --v_groupsize 128 \
#     --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
#     --lwc --lac --cali_trans --add_diag \
#     --output_dir ./outputs --save_matrix \
#     --epochs 15

# W4A16
# python ./main.py \
#     --model ./modelzoo/llama-3/llama-3-8b \
#     --w_bits 4 \
#     --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
#     --lwc --lac --cali_trans --add_diag \
#     --output_dir ./outputs --exp_name wonly --save_matrix \
#     --lm_eval --lm_eval_batch_size 16
