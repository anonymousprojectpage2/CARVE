#!/bin/bash

# ./bash_scripts/continual_merge_libero.sh MergeVLA
# ./bash_scripts/continual_merge_libero.sh OpenVLA
# ./bash_scripts/continual_merge_libero.sh VLAAdapter

MODEL_TYPE="${1:-OpenVLA}"  
echo "$MODEL_TYPE"

if [ "$MODEL_TYPE" == "MergeVLA" ]; then
    export CKPT_SPATIAL="./MergeVLA/MergeVLA_LIBERO_downloaded/Spatial"
    export CKPT_OBJECT="./MergeVLA/MergeVLA_LIBERO_downloaded/Object"
    export CKPT_GOAL="./MergeVLA/MergeVLA_LIBERO_downloaded/Goal"
    export CKPT_10="./MergeVLA/MergeVLA_LIBERO_downloaded/Long10"

    export SAVE_DIR="./MergeVLA/continual_merged_models/MergeVLA/LIBERO"

    export TASK_ORDER="spatial,object,goal,10"
    export TALL_MASK_LAMBDA="0.6"
    export K_GATE="8"
    export ACTION_HEAD_LAYER_NUM="1"
    export RESUME="False"
    export NOTE="AHnum_1_k_8"
    SCRIPT="model_merging/continual_mergy_MergeVLA.py"

elif [ "$MODEL_TYPE" == "OpenVLA" ]; then
    export BASE_CHECKPOINT="openvla/openvla-7b"

    export CKPT_SPATIAL="openvla/openvla-7b-finetuned-libero-spatial"
    export CKPT_OBJECT="openvla/openvla-7b-finetuned-libero-object"
    export CKPT_GOAL="openvla/openvla-7b-finetuned-libero-goal"
    export CKPT_10="openvla/openvla-7b-finetuned-libero-10"

    export SAVE_DIR="./continual_merged_models/OpenVLA/LIBERO"

    export TASK_ORDER="spatial,object,goal,10"
    export RESUME="False"
    export NOTE=""
    SCRIPT="model_merging/continual_mergy.py"

elif [ "$MODEL_TYPE" == "VLAAdapter" ]; then
    export BASE_CHECKPOINT="./MergeVLA/VLA-Adapter/pretrained_models/vla_config"

    export CKPT_SPATIAL="VLA-Adapter/LIBERO-Spatial-Pro"
    export CKPT_OBJECT="VLA-Adapter/LIBERO-Object-Pro"
    export CKPT_GOAL="VLA-Adapter/LIBERO-Goal-Pro"
    export CKPT_10="VLA-Adapter/LIBERO-Long-Pro"

    export SAVE_DIR="./continual_merged_models/VLAAdapter/LIBERO"
    
    export TASK_ORDER="spatial,object,goal,10"
    export RESUME="False"
    export NOTE=""
    SCRIPT="model_merging/continual_mergy.py"
fi

python $SCRIPT