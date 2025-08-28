## Automatic Cancer Segmentation and Classification in 3D CT Scans

This repository contains the code for training a UNet-based joint classification + segmentation model for pancreatic lesion segmentation and classification in 3D CT scans. 

### Methods

* The network architecture is based on the nnUNetResEncM preset provided by the nnUNet framework. The existing encoder-decoder structure is kept intact, a classifier head to the output of the encoder is attached. The classifier is 3-layer MLP with 256 and 128 hidden units and 3 output neurons for the 3-way lesion subtype classification. The model architecture can be found  in `models/unet.py`. The model components in detail can be found in `models/network.json`.

* We used the [MetricsReloaded](https://github.com/Project-MONAI/MetricsReloaded) package for evaluation. Specifically, the segmentation accuracy of the pancreas and the lesion was evaluated using Dice score and the lesion subtype classification accuracy was evaluated using macro-average F1 score from `scipy`'s `classification_report` function. The evaluation code can be found in `evaluation/compute_metrics.py`.

* We used the 2D nnUNet model trained on patches of size [128, 192] on images resampled to high isotropic resolution of [0.73 x 0.73]. It was trained for 100 epochs only with the SGD optimizer and an initial learning rate of $0.01$ and a polynomial LR scheduler. More details about the preprocessing, training can be found in the associated report.


### Results

Both the default trainer and the trainer with extensive data augmentations (\texttt{DA5}) performed similarly on both segmentation and classification accuracy.

| **Model**                    | **DSC** | | **F1-macro** |
|---------------------- |-------------|-----------|---------------------|
|                       | label=1  | label=2               |      |
| nnUNetResEncM_default | 0.80     | 0.46                  | 0.78 |
| nnUNetResEncM_DA5     | 0.81     | 0.46                 | 0.76 |


### How to run

The code was developed and tested on Ubuntu, with Python 3.10 and CUDA 12.0. Models were trained on a single NVIDIA A6000 GPU with 48GB memory.

1. Create a conda environment and install the required packages listed below.

    ```bash 
   conda create -n flare_multitask python=3.10 -y
   conda activate flare_multitask
    ```
2. Install requirements
    
    ```bash
   pip install -r requirements.txt
    ```

3. Install these packages from source:
    - [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
    - [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
    - [MetricsReloaded](https://github.com/ivadomed/MetricsReloaded)


4. To run preprocessing, training and inference, run `bash scripts/run_nnunet.sh` after ensuring that all variables are set correctly. 