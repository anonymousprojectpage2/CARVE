ckpt=./continual_merged_models/order_spatial_object_goal_10_AHnum_1_k_8/step_3
tasks=("libero_spatial" "libero_object" "libero_goal" "libero_10")

LOG_FILE= $ckpt
exec > >(tee -a $LOG_FILE) 2>&1

# MergeVLA
num_trial=50
for task in "${tasks[@]}"; do
  python ./experiments/robot/libero/run_libero_eval.py \
    --num_images_in_input 2 \
    --pretrained_checkpoint $ckpt \
    --task_suite_name $task \
    --load_moe True \
    --pretrained_vlm_checkpoint ./pretrained_models/Pretrained-VLM \
    --k_gate 8 \
    --action_head_layer_num 1 \
    --num_trials_per_task $num_trial \
    --save_rollout False
done

# OpenVLA
for task in "${tasks[@]}"; do
    python ./openvla/experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint $ckpt \
        --task_suite_name $task \
        --num_trials_per_task 50 \
        --save_rollout False
done

# VLA-Adapter_Plus
for task in "${tasks[@]}"; do
    python ./experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint $ROOT_CKPT \
        --task_suite_name $task \
        --num_trials_per_task 50 \
        --save_rollout False
done