#!/bin/bash

# run inference on all subjects with and without TTA


PATH_ROOT="/home/GRAMES.POLYMTL.CA/u114716/wanglab_ml_quiz"
PATH_REPO="${PATH_ROOT}/flare_challenge_multitask"
PATH_IMAGES="${PATH_ROOT}/nnUNet_raw/Dataset601_FlareMultiTask/imagesVal"
trainer_folder="nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__2d" 

csv_without_tta="${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder}/inference_noTTA/wo_tta.csv"
csv_with_tta="${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder}/inference_TTA/w_tta.csv"


# iterate over all subjects in the images folder
for img in ${PATH_IMAGES}/*_0000.nii.gz; do
    echo "-------------------------------------------"
    echo "Running inference withOUT TTA on image: ${img}"
    echo "-------------------------------------------"

    pred_img=$(basename ${img} | sed 's/_0000.nii.gz/_pred.nii.gz/')

    # run without TTA
    all_output=$(
    python ${PATH_REPO}/testing/run_inference_single_subject.py \
        -i ${img} \
        -o ${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder}/inference_noTTA \
        -path-model ${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder} \
        -use-best-checkpoint
    )
    # get only total_time line
    inference_time=$(echo "${all_output}" | grep "total_time" | sed 's/total_time: //')
    # append to csv
    echo "${pred_img},${inference_time}" >> ${csv_without_tta}

done


for img in ${PATH_IMAGES}/*_0000.nii.gz; do
    pred_img=$(basename ${img} | sed 's/_0000.nii.gz/_pred.nii.gz/')

    echo ""
    echo "-------------------------------------------"
    echo "Running inference with TTA on image: ${img}"
    echo "-------------------------------------------"

    # run with TTA
    all_output=$(
    python ${PATH_REPO}/testing/run_inference_single_subject.py \
        -i ${img} \
        -o ${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder}/inference_TTA \
        -path-model ${PATH_ROOT}/nnUNet_results/Dataset601_FlareMultiTask/${trainer_folder} \
        -use-best-checkpoint \
        -use-mirroring
    )

    # get only total_time line
    inference_time=$(echo "${all_output}" | grep "total_time" | sed 's/total_time: //')
    # append to csv
    echo "${pred_img},${inference_time}" >> ${csv_with_tta}
    echo ""
done






