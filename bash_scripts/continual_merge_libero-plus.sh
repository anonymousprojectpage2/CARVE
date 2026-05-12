#!/bin/bash
MODEL_TYPE="MergeVLA"

if [ "$MODEL_TYPE" == "MergeVLA" ]; then
    export CKPT_SPATIAL="./experts/libero_plus/MergeVLA/spatial"
    export CKPT_OBJECT="./experts/libero_plus/MergeVLA/object"
    export CKPT_GOAL="./experts/libero_plus/MergeVLA/goal"
    export CKPT_10="./experts/libero_plus/MergeVLA/10"

    export SAVE_DIR="./continual_merged_models/MergeVLA/LIBERO_Plus"

    export TASK_ORDER="spatial,object,goal,10"
    export TALL_MASK_LAMBDA="0.6"
    export K_GATE="8"
    export ACTION_HEAD_LAYER_NUM="1"
    export RESUME="False"
    export NOTE="AHnum_1_k_8"
    SCRIPT="model_merging/continual_mergy_MergeVLA.py"


elif [ "$MODEL_TYPE" == "OpenVLA" ]; then
    export BASE_CHECKPOINT="openvla/openvla-7b"

    export CKPT_SPATIAL="./experts/libero_plus/OpenVLA/spatial"
    export CKPT_OBJECT="./experts/libero_plus/OpenVLA/object"
    export CKPT_GOAL="./experts/libero_plus/OpenVLA/goal"
    export CKPT_10="./experts/libero_plus/OpenVLA/10"

    export SAVE_DIR="./continual_merged_models/OpenVLA/LIBERO_Plus"

    export RESUME="False"
    export NOTE=""
    SCRIPT="model_merging/continual_mergy.py"


elif [ "$MODEL_TYPE" == "VLAAdapter" ]; then
    export BASE_CHECKPOINT="/shared/kyungjin/MergeVLA/VLA-Adapter/pretrained_models/vla_config"

    export CKPT_SPATIAL="./experts/libero_plus/VLAAdapter/spatial"
    export CKPT_OBJECT="./experts/libero_plus/VLAAdapter/object"
    export CKPT_GOAL="./experts/libero_plus/VLAAdapter/goal"
    export CKPT_10="./experts/libero_plus/VLAAdapter/10"

    export SAVE_DIR="./continual_merged_models/VLAAdapter/LIBERO_Plus"

    export RESUME="False"
    export NOTE=""
    SCRIPT="model_merging/continual_mergy.py"
fi

python $SCRIPT