"""
Convert BIDS-structured SCI datasets (sci-zurich, sci-colorado, dcm-zurich-lesions, dcm-zurich-lesions-20231115, etc.) to the nnUNetv2
REGION-BASED and MULTICHANNEL training format depending on the input arguments.

dataset.json:

```json
    "channel_names": {
        "0": "acq-ax_T2w"
    },
    "labels": {
        "background": 0,
        "sc": [
            1,
            2
        ],
        "lesion": 2
    },
    "regions_class_order": [
        1,
        2
    ],
```

Full details about the format can be found here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Note: the script performs RPI reorientation of the images and labels

Usage example single dataset:
    python convert_bids_to_nnUNetv2_region-based.py
        --path-data ~/data/dcm-zurich-lesions
        --path-out ${nnUNet_raw}
        -dname DCMlesions
        -dnum 601
        --split 0.8 0.2
        --seed 50
        --region-based

Authors: Naga Karthik, Jan Valosek
"""

import argparse
from pathlib import Path
import json
import os
import re
import shutil
import yaml
from collections import OrderedDict
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import Image
from tqdm import tqdm
import random
import nibabel as nib
import numpy as np



def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 REGION-BASED format.')
    parser.add_argument('--path-data', required=True, type=str, help='Path to dataset.')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='DCMlesionsRegionBased', type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', default=601, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--debug', action='store_true', help='Debug mode with limited number of images.')
    return parser


def find_subtype_in_path(path):
    """Extracts lesion subtype identifier from the given path.

    Args:
    path (str): Input path containing a subtype identifier.

    Returns:
    str: Extracted subtype identifier or None if not found.
    """
    # Find 'dcm-zurich-lesions' or 'dcm-zurich-lesions-20231115'
    if 'dcm' in path:
        match = re.search(r'dcm-zurich-lesions(-\d{8})?', path)
    elif 'sci' in path:
        match = re.search(r'sci-zurich|sci-colorado|sci-paris', path)
    elif 'site' in path:
        # NOTE: PRAXIS data has 'site-xxx' in the path (and doesn't have the site names themselves in the path)
        match = re.search(r'site-\d{3}', path)

    return match.group(0) if match else None


def create_yaml(train_niftis, test_nifitis, val_niftis, path_out, args, train_ctr, test_ctr, val_ctr):
    # create a yaml file containing the list of training and test niftis
    niftis_dict = {
        f"train": sorted(train_niftis),
        f"val": sorted(val_niftis),
        f"test": sorted(test_nifitis)
    }

    # write the train and test niftis to a yaml file
    with open(os.path.join(path_out, f"train_test_split.yaml"), "w") as outfile:
        yaml.dump(niftis_dict, outfile, default_flow_style=False)

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['numTraining'] = train_ctr
    json_dict['numValidation'] = val_ctr
    json_dict['numTest'] = test_ctr
    json_dict['image_orientation'] = "RPI"

    # The following keys are the most important ones.
    json_dict['channel_names'] = {
        0: "CT",
    }

    json_dict['labels'] = {
        "background": 0,
        "pancreas": 1,
        "lesion": 2,
    }

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


def main():
    parser = get_parser()
    args = parser.parse_args()

    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_imagesVal = Path(os.path.join(path_out, 'imagesVal'))
    path_out_labelsVal = Path(os.path.join(path_out, 'labelsVal'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))

    # create directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesVal).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsVal).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)

    # save output to a log file
    logger.add(os.path.join(path_out, "logs.txt"), rotation="10 MB", level="INFO")

    # Check if dataset paths exist
    if not os.path.exists(args.path_data):
        raise ValueError(f"Path {args.path_data} does not exist.")

    # # Get sites from the input paths
    # sites = set(find_subtype_in_path(path) for path in args.path_data if find_subtype_in_path(path))

    train_images, test_images, val_images = {}, {}, {}

    # loop over the datasets
    root = Path(args.path_data)  # assuming all datasets have the same structure

    # get recursively all GT '_label-lesion' files
    image_files = [str(path) for path in root.rglob(f'*_0000.nii.gz')]
    for file in image_files:
        if 'train' in file:
            train_images[os.path.basename(file)] = file
        elif 'validation' in file:
            val_images[os.path.basename(file)] = file
        elif 'test' in file:
            test_images[os.path.basename(file)] = file

    # keep only 10 images in training set for quick testing
    if args.debug:
        # randomly sample 15 train, 5 val, 5 test images
        train_images = dict(random.sample(list(train_images.items()), k=15))
        val_images = dict(random.sample(list(val_images.items()), k=5))
        test_images = dict(random.sample(list(test_images.items()), k=5))

    all_images = list(train_images.values()) + list(test_images.values()) + list(val_images.values())

    logger.info(f"Found subjects in the training set: {len(train_images)}")
    logger.info(f"Found subjects in the validation set: {len(val_images)}")
    logger.info(f"Found subjects in the test set: {len(test_images)}")

    # Counters for train and test sets
    train_ctr, test_ctr, val_ctr = 0, 0, 0
    train_niftis, test_nifitis, val_niftis = [], [], []
    # Loop over all images
    for subject_image_file in tqdm(all_images, desc="Iterating over all images"):

        # Construct path to the background image
        subject_label_file = subject_image_file.replace('_0000.nii.gz', '.nii.gz')

        # Train images
        if subject_image_file in train_images.values():

            train_ctr += 1
            # add the subject image file to the list of training niftis
            train_niftis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('_0000.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                        f"{args.dataset_name}_{sub_name}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                    f"{args.dataset_name}_{sub_name}.nii.gz")

        # Validation images
        elif subject_image_file in val_images.values():
            
            val_ctr += 1
            # add the image file to the list of validation niftis
            val_niftis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('_0000.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(path_out_imagesVal,
                                                        f"{args.dataset_name}_{sub_name}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsVal,
                                                    f"{args.dataset_name}_{sub_name}.nii.gz")

            # # copy the files to new structure
            # shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            # shutil.copyfile(subject_label_file, subject_label_file_nnunet)

        # Test images
        elif subject_image_file in test_images.values():

            test_ctr += 1
            # add the image file to the list of testing niftis
            test_nifitis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('_0000.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(Path(path_out_imagesTs, 
                                                          f'{args.dataset_name}_{sub_name}_0000.nii.gz'))

            # # copy the files to new structure
            # shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            # # shutil.copyfile(subject_label_file, subject_label_file_nnunet)
            # # print(f"\nCopying {subject_image_file} to {subject_image_file_nnunet}")

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject_image_file)

        # copy the files to new structure
        shutil.copyfile(subject_image_file, subject_image_file_nnunet)
        
        # convert the image and label to RPI using the Image class
        image = Image(subject_image_file_nnunet)
        image.change_orientation("RPI")
        image.save(subject_image_file_nnunet)

        if 'test' not in subject_image_file:  # only copy labels for train and val sets
            shutil.copyfile(subject_label_file, subject_label_file_nnunet)
            label = Image(subject_label_file_nnunet)
            label.change_orientation("RPI")
            label.change_type(dtype='int16')
            label.save(subject_label_file_nnunet)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (across all sites): {train_ctr}")
    logger.info(f"Number of validation images (across all sites): {val_ctr}")
    logger.info(f"Number of test images (across all sites): {test_ctr}")

    # create the yaml file containing the train and test niftis
    create_yaml(train_niftis, test_nifitis, val_niftis, path_out, args, train_ctr, test_ctr, val_ctr)


if __name__ == "__main__":
    main()