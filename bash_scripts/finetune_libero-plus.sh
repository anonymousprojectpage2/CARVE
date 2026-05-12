# MergeVLA / OpenVLA / VLAAdapter
MODEL_TYPE="MergeVLA"
OUTPUT_DIR=./experts/libero_plus/$MODEL_TYPE

if [ "$MODEL_TYPE" == "MergeVLA" ]; then
    declare -A TASK_CKPTS=(
        ["libero_spatial"]="./MergeVLA/MergeVLA_LIBERO_downloaded/Spatial"
        ["libero_object"]="./MergeVLA/MergeVLA_LIBERO_downloaded/Object"
        ["libero_goal"]="./MergeVLA/MergeVLA_LIBERO_downloaded/Goal"
        ["libero_10"]="./MergeVLA/MergeVLA_LIBERO_downloaded/Long10"
    )
elif [ "$MODEL_TYPE" == "OpenVLA" ]; then
    declare -A TASK_CKPTS=(
        ["libero_spatial"]="openvla/openvla-7b-finetuned-libero-spatial"
        ["libero_object"]="openvla/openvla-7b-finetuned-libero-object"
        ["libero_goal"]="openvla/openvla-7b-finetuned-libero-goal"
        ["libero_10"]="openvla/openvla-7b-finetuned-libero-10"
    )
elif [ "$MODEL_TYPE" == "VLAAdapter" ]; then
    declare -A TASK_CKPTS=(
        ["libero_spatial"]="VLA-Adapter/LIBERO-Spatial-Pro"
        ["libero_object"]="VLA-Adapter/LIBERO-Object-Pro"
        ["libero_goal"]="VLA-Adapter/LIBERO-Goal-Pro"
        ["libero_10"]="VLA-Adapter/LIBERO-Long-Pro"
    )
fi

declare -A DATA_FOLDERS=(
    ["libero_spatial"]="libero_spatial"
    ["libero_object"]="libero_object"
    ["libero_goal"]="libero_goal"
    ["libero_10"]="libero_10"
)

for data_name in "libero_spatial" "libero_object" "libero_goal" "libero_10"; do
    ckpt=${TASK_CKPTS[$data_name]}
    data_folder=${DATA_FOLDERS[$data_name]}
    current_time=$(date "+%Y%m%d-%H%M%S")

    echo "=========================================="
    echo "Fine-tuning: $data_name"
    echo "Base model: $ckpt"
    echo "=========================================="

    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
        ./vla-scripts/finetune_libero_plus.py \
        --vlm_path $ckpt \
        --config_file_path ./pretrained_models/configs \
        --data_root_dir $LIBERO_PLUS_DATA \
        --dataset_name $data_name \
        --run_root_dir $OUTPUT_DIR \
        --use_film False \
        --num_images_in_input 2 \
        --use_proprio True \
        --use_lora True \
        --use_fz False \
        --use_minivlm True \
        --image_aug True \
        --num_steps_before_decay 10000 \
        --max_steps 10005 \
        --save_freq 10000 \
        --save_latest_checkpoint_only False \
        --merge_lora_during_training True \
        --batch_size 8 \
        --grad_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --lora_rank 64 \
        --run_id_note MergeVLA--libero_plus--$data_name--$current_time

    echo "Finished: $data_name"
done

echo "All tasks done!"