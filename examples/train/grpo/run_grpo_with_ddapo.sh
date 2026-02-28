# Minimal GRPO run with DDAPO attention reward (no vLLM)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MAX_PIXELS=602112 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs ddapo_attention external_r1v_acc format \
    --reward_weight 1.0 4.0 1.0 \
    --use_vllm false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset /home/lujunxi57/ms-swift-rlvr/data/virl39k/train.jsonl \
    --overlong_filter false \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --dataloader_num_workers 16 \
    --num_generations 4 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --async_generate false \
    --beta 0 \
    --loss_type grpo \
    --advantage_estimator grpo
