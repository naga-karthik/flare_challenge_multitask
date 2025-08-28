#!/bin/bash
#
# Training nnUNetv2 on multiple folds
#
#

# define arguments for nnUNet
dataset_name="Dataset601_FlareMultiTask"
dataset_num="601"
# nnunet_trainer="nnUNetTrainerDiceCELoss_noSmooth"         # default: nnUNetTrainer or nnUNetTrainer_2000epochs
# nnunet_trainer="nnUNetTrainerDA5"                           # trainer variant with aggressive data augmentation
# nnunet_trainer="nnUNetTrainerDA5_DiceCELoss_noSmooth"       # custom trainer
nnunet_trainer="nnUNetTrainer_100epochs"                    # options: nnUNetTrainer_1epoch, nnUNetTrainer_5epochs

model_type="M"                                              # options: "M" or "L" or "XL"
nnunet_planner="nnUNetPlannerResEnc${model_type}"           # default: nnUNetPlannerResEncM/L
nnunet_plans_file="nnUNetResEncUNet${model_type}Plans"

# configurations=("3d_fullres" "2d")                        # for 2D training, use "2d"
configurations=("2d" "3d_fullres")                        # for 2D training, use "2d"
cuda_visible_devices=3


# Select number of folds here
# folds=(4 3 2 1 0)
folds=(2 0 1 3 4)
# folds=(2 0)


PATH_ROOT_FOLDER="/home/GRAMES.POLYMTL.CA/u114716/wanglab_ml_quiz"
# temporarily export the nnUNet environment variables
export nnUNet_raw=${PATH_ROOT_FOLDER}/nnUNet_raw
export nnUNet_preprocessed=${PATH_ROOT_FOLDER}/nnUNet_preprocessed
export nnUNet_results=${PATH_ROOT_FOLDER}/nnUNet_results

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
if [[ $model_type == "M" ]]; then
    nnUNetv2_plan_and_preprocess -d ${dataset_num} -pl ${nnunet_planner} --verify_dataset_integrity
else
    nnUNetv2_plan_and_preprocess -d ${dataset_num} --verify_dataset_integrity
fi
# exit 0

for config in "${configurations[@]}"; do

    for fold in ${folds[@]}; do
        
        echo "-------------------------------------------"
        echo "Training on configuration: ${config}, Fold: $fold"
        echo "-------------------------------------------"

        # training
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} ${config} ${fold} -tr ${nnunet_trainer} -p ${nnunet_plans_file}

        echo ""
        echo "-------------------------------------------"
        echo "Training completed, Testing on Fold $fold"
        echo "-------------------------------------------"
        # exit 0

        # inference
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesVal -p ${nnunet_plans_file} -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${config}/fold_${fold}/test -d ${dataset_num} -f ${fold} -c ${config}

        echo ""
        echo "-------------------------------------------"
        echo " Inference completed on Fold $fold"
        echo "-------------------------------------------"
    done
done