"""
This script is used to run inference on a single subject using a nnUNetV2 model.

Example spinal cord segmentation:
    python run_inference_single_subject.py
        -i <image>.nii.gz
        -o /path/to/output/folder
        -path-model /path/to/model
        -use-best-checkpoint
        -use-gpu
"""


import os
import shutil
import subprocess
import argparse
import datetime

import torch
import glob
import time
import tempfile

# from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNet model.')
    parser.add_argument('-i', help='Input image to segment. Example: sub-001_T2w.nii.gz', required=True)
    parser.add_argument('-o', help='Output filename. Example: sub-001_T2w_seg_nnunet.nii.gz', required=True)
    parser.add_argument('-path-model', help='Path to the model directory. This folder should contain individual '
                                            'folders like fold_0, fold_1, etc. and dataset.json, '
                                            'dataset_fingerprint.json and plans.json files.', required=True, type=str)
    parser.add_argument('-use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('-use-mirroring', action='store_true', default=False,
                        help='Use mirroring for test time augmentation. Default: False')
    parser.add_argument('-use-best-checkpoint', action='store_true', default=False,
                        help='Use the best checkpoint (instead of the final checkpoint) for prediction. '
                             'NOTE: nnUNet by default uses the final checkpoint. Default: False')
    parser.add_argument('-tile-step-size', default=0.5, type=float,
                        help='Tile step size defining the overlap between images patches during inference. '
                             'Default: 0.5 '
                             'NOTE: changing it from 0.5 to 0.9 makes inference faster but there is a small drop in '
                             'performance.')
    return parser


def tmp_create():
    """
    Create temporary folder and return its path
    """
    prefix = f"flareCT_prediction_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Creating temporary folder ({tmpdir})")
    return tmpdir


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.
    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    Taken (shamelessly) from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension. Taken (shamelessly) from:
    https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    :param fname: absolute or relative file name. Example: t2.nii.gz
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii
    Examples:
    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)



def main():
    parser = get_parser()
    args = parser.parse_args()

    fname_file = args.i
    path_out = args.o
    # fname_file_out = args.o
    print(f'Found {fname_file} file.')

    # # Create temporary directory in the temp to store the reoriented images
    # tmpdir = tmp_create()
    # # Copy the file to the temporary directory using shutil.copyfile
    # fname_file_tmp = os.path.join(tmpdir, os.path.basename(fname_file))
    # shutil.copyfile(fname_file, fname_file_tmp)
    # print(f'Copied {fname_file} to {fname_file_tmp}')

    # NOTE: for individual images, the _0000 suffix is not needed.
    # BUT, the images should be in a list of lists
    fname_file_tmp_list = [[fname_file]]

    # Use all the folds available in the model folder by default
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]
    # folds_avail = [2]

    # # Create directory for nnUNet prediction
    # tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    # fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(fname_file_tmp, '_pred')))
    # os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...')
    start = time.time()

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,                      # applies gaussian noise and gaussian blur
        use_mirroring=True if args.use_mirroring else False, # test time augmentation by mirroring on all axes
        perform_everything_on_device=True if args.use_gpu else False,
        device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('Running inference on device: {}'.format(predictor.device))

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(args.path_model),
        use_folds=folds_avail,
        checkpoint_name='checkpoint_final.pth' if not args.use_best_checkpoint else 'checkpoint_best.pth',
    )
    print('Model loaded successfully. Fetching test data...')

    # NOTE: for individual files, the image should be in a list of lists
    predictor.predict_from_files(
        list_of_lists_or_source_folder=fname_file_tmp_list,
        output_folder_or_list_of_truncated_output_files=path_out,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    end = time.time()
    print('Inference done.')
    
    total_time = end - start
    print(f"total_time: {int(total_time)}")

    print('Total inference time: {} minute(s) {} seconds'.format(int(total_time // 60), int(round(total_time % 60))))

    # # Copy .nii.gz file from tmpdir_nnunet to tmpdir
    # pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
    # shutil.copyfile(pred_file, fname_prediction)
    # print(f"fname_file_out: {fname_file_out}")
    # print(fname_prediction)

    # # move the prediction file to the output location
    # print(f'Moving {fname_prediction} to {fname_file_out}')
    # shutil.move(fname_prediction, fname_file_out)
    
    # print('Deleting the temporary folder...')
    # # Delete the temporary folder
    # shutil.rmtree(tmpdir)

    # print('-' * 50)
    # print(f'Created {fname_file_out}')
    # print('-' * 50)


if __name__ == '__main__':
    main()