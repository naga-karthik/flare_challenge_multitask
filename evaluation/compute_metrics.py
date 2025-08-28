"""
Compute MetricsReloaded metrics for segmentation and classification tasks. 
Given path to the model folder containing all folds, the script computes metrics for each fold and outputs a csv
containing average metrics across all folds.
The metrics are computed for each unique label (class) in the reference (ground truth) image.

Usage example:
python compute_metrics.py -path-model /path/to/model/folder -reference /path/to/reference/folder -output metrics.csv

reference   prediction	label	dsc nsd	EmptyRef	EmptyPred
seg.nii.gz	pred.nii.gz	1.0	0.819	0.945   False	False
seg.nii.gz	pred.nii.gz	2.0	0.743	0.923   False	False

"""


import os
import re
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics import classification_report

from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM


# This dictionary is used to rename the metric columns in the output CSV file
METRICS_TO_NAME = {
    'dsc': 'DiceSimilarityCoefficient',
}


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute MetricsReloaded metrics for segmentation tasks.')

    # Arguments for model, data, and training
    parser.add_argument('-path-model', required=True, type=str,
                        help='Path to the folder containing all folds (i.e. fold_0, fold_1, etc)')
    parser.add_argument('-reference', required=True, type=str,
                        help='Path to the folder with nifti images of reference (ground truth) or path to a single '
                             'nifti image of reference (ground truth).')
    parser.add_argument('-metrics', nargs='+', required=False,
                        default=['dsc', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error',
                                 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score',
                                 'ref_count', 'pred_count', 'lcwa'],
                        help='List of metrics to compute. For details, '
                             'see: https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/metrics.html.')
    parser.add_argument('-output', type=str, default='metrics.csv', required=False,
                        help='Path to the output CSV file to save the metrics. Default: metrics.csv')

    return parser


def load_nifti_image(file_path):
    """
    Construct absolute path to the nifti image, check if it exists, and load the image data.
    :param file_path: path to the nifti image
    :return: nifti image data
    """
    file_path = os.path.expanduser(file_path)   # resolve '~' in the path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    nifti_image = nib.load(file_path)
    return nifti_image.get_fdata()


def fetch_overlapping_filename(filename_path, prefix='FlareMultiTask_'):
    """
    Use regex to match entire filename of this format FlareMultiTask_quiz_0_168
    """
    filename = os.path.basename(filename_path)
    pattern = re.compile(rf'({prefix}\w+_\w+_\w+)')
    match = pattern.match(filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'Filename {filename} does not match the expected format.')

def get_images(prediction, reference):
    """
    Get all files (predictions and references/ground truths) in the input directories.
    :param prediction: path to the directory with prediction files
    :param reference: path to the directory with reference (ground truth) files
    :return: list of prediction files, list of reference/ground truth files
    """
    # Get all files in the directories
    prediction_files = [os.path.join(prediction, f) for f in os.listdir(prediction) if f.endswith('.nii.gz')]
    reference_files = [os.path.join(reference, f) for f in os.listdir(reference) if f.endswith('.nii.gz')]

    if not prediction_files:
        raise FileNotFoundError(f'No prediction files found in {prediction}.')
    if not reference_files:
        raise FileNotFoundError(f'No reference (ground truths) files found in {reference}.')

    # Create dataframe for prediction_files
    df_pred = pd.DataFrame(prediction_files, columns=['filename'])
    df_pred['fname'] = df_pred['filename'].apply(fetch_overlapping_filename)

    # Create dataframe for reference_files with participant_id, acq_id, run_id
    df_ref = pd.DataFrame(reference_files, columns=['filename'])
    df_ref['fname'] = df_ref['filename'].apply(fetch_overlapping_filename)

    # Merge the two dataframes on participant_id, acq_id, run_id
    df = pd.merge(df_pred, df_ref, on=['fname'], how='outer', suffixes=('_pred', '_ref'))
    # Drop 'fname'
    df.drop(['fname'], axis=1, inplace=True)
    # # Drop rows with NaN values. In other words, keep only the rows where both prediction and reference files exist
    # df.dropna(inplace=True)

    prediction_files = df['filename_pred'].tolist()
    reference_files = df['filename_ref'].tolist()

    return prediction_files, reference_files

def compute_f1_macro(path_csv):
    """
    Compute F1-macro score for classification task
    :param path_csv: path to the CSV file with the classification results
    :return: F1-macro score
    """
    # check if file exists
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f'File {path_csv} does not exist.')

    df = pd.read_csv(path_csv, header=None)
    df.columns = ['name', 'subtype']

    y_true, y_pred = [], []
    for i, row in df.iterrows():
        y_true.append(int(row['name'].split('_')[-2]))
        y_pred.append(int(row['subtype']))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Compute classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']

    return f1_macro

def compute_metrics_single_subject(prediction, reference, metrics):
    """
    Compute MetricsReloaded metrics for a single subject
    :param prediction: path to the nifti image with the prediction
    :param reference: path to the nifti image with the reference (ground truth)
    :param metrics: list of metrics to compute
    """
    # load nifti images
    dirname_pred = f"{os.path.basename(os.path.dirname(prediction))}/{os.path.basename(prediction)}"
    dirname_ref = f"{os.path.basename(os.path.dirname(reference))}/{os.path.basename(reference)}"

    print(f'\nProcessing:\n\tPrediction: {dirname_pred}\n\tReference: {dirname_ref}')
    prediction_data = load_nifti_image(prediction)
    reference_data = load_nifti_image(reference)

    # check whether the images have the same shape and orientation
    if prediction_data.shape != reference_data.shape:
        raise ValueError(f'The prediction and reference (ground truth) images must have the same shape. '
                         f'The prediction image has shape {prediction_data.shape} and the ground truth image has '
                         f'shape {reference_data.shape}.')

    # get all unique labels (classes)
    # for example, for nnunet region-based segmentation, spinal cord has label 1, and lesions have label 2
    unique_labels_reference = np.unique(reference_data)
    unique_labels_reference = unique_labels_reference[unique_labels_reference != 0]  # remove background
    unique_labels_prediction = np.unique(prediction_data)
    unique_labels_prediction = unique_labels_prediction[unique_labels_prediction != 0]  # remove background

    # Get the unique labels that are present in the reference OR prediction images
    unique_labels = np.unique(np.concatenate((unique_labels_reference, unique_labels_prediction)))

    # append entry into the output_list to store the metrics for the current subject
    metrics_dict = {'reference': reference, 'prediction': prediction}

    # loop over all unique labels, e.g., voxels with values 1, 2, ...
    # by doing this, we can compute metrics for each label separately, e.g., separately for spinal cord and lesions
    for label in unique_labels:
        # create binary masks for the current label
        if label == 1:
            # to compute the metric on whole pancreas (normal + lesion), we need to sum the two labels
            prediction_data_label = (prediction_data > 0.5).astype(float)
            reference_data_label = (reference_data > 0.5).astype(float)

        else:
            prediction_data_label = np.array(prediction_data == label, dtype=float)
            reference_data_label = np.array(reference_data == label, dtype=float)

        bpm = BPM(prediction_data_label, reference_data_label, measures=metrics)
        dict_seg = bpm.to_dict_meas()
        # Store info whether the reference or prediction is empty
        dict_seg['EmptyRef'] = bpm.flag_empty_ref
        dict_seg['EmptyPred'] = bpm.flag_empty_pred
        # add the metrics to the output dictionary
        metrics_dict[label] = dict_seg

    # Special case when both the reference and prediction images are empty
    else:
        label = 1
        bpm = BPM(prediction_data, reference_data, measures=metrics)
        dict_seg = bpm.to_dict_meas()

        # Store info whether the reference or prediction is empty
        dict_seg['EmptyRef'] = bpm.flag_empty_ref
        dict_seg['EmptyPred'] = bpm.flag_empty_pred
        # add the metrics to the output dictionary
        metrics_dict[label] = dict_seg

    return metrics_dict


def build_output_dataframe(output_list):
    """
    Convert JSON data to pandas DataFrame
    :param output_list: list of dictionaries with metrics
    :return: pandas DataFrame
    """
    rows = []
    for item in output_list:
        # Extract all keys except 'reference' and 'prediction' to get labels (e.g. 1.0, 2.0, etc.) dynamically
        labels = [key for key in item.keys() if key not in ['reference', 'prediction']]
        for label in labels:
            metrics = item[label]  # Get the dictionary of metrics
            # Dynamically add all metrics for the label
            row = {
                "reference": item["reference"],
                "prediction": item["prediction"],
                "label": label,
            }
            # Update row with all metrics dynamically
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def main():
    # parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    folds_avail = [f for f in os.listdir(args.path_model) if f.startswith('fold_')]

    f1_mac_avg = []
    df_folds = pd.DataFrame()
    for fold in sorted(folds_avail):

        print(f"Computing metrics for {fold} ...")
        path_predictions = os.path.join(args.path_model, fold, 'test')

        # ----------------------------
        # compute f1 macro
        # ----------------------------
        f1_macro_classification = compute_f1_macro(os.path.join(path_predictions, "lesion_subtype_predictions.csv"))
        f1_mac_avg.append(f1_macro_classification)
        print(f"F1-macro score for lesion subtype classification on {fold}: {f1_macro_classification:.3f}")

        # Initialize a list to store the output dictionaries (representing a single reference-prediction pair per subject)
        output_list = list()

        # Args.prediction and args.reference are paths to folders with multiple nii.gz files (i.e., MULTIPLE subjects)
        if os.path.isdir(path_predictions) and os.path.isdir(args.reference):
            # Get all files in the directories
            prediction_files, reference_files = get_images(path_predictions, args.reference)

            for pred, ref in zip(prediction_files, reference_files):
                # print(f'Prediction: {pred}, Reference: {ref}')
                metrics_dict = compute_metrics_single_subject(pred, ref, args.metrics)
                # Append the output dictionary (representing a single reference-prediction pair per subject)
                output_list.append(metrics_dict)
        # else:
        #     metrics_dict = compute_metrics_single_subject(args.prediction, args.reference, args.metrics,)
        #     # Append the output dictionary (representing a single reference-prediction pair per subject) to the output_list
        #     output_list.append(metrics_dict)


        # Convert JSON data to pandas DataFrame
        df = build_output_dataframe(output_list)

        df['fold'] = fold
        # df.to_csv(os.path.join(args.path_model, f"metrics_all_subjects_{fold}.csv"), index=False)
        df_folds = pd.concat([df_folds, df], axis=0)

    # Average F1-macro score across all folds
    f1_mac_mean, f1_mac_std = np.mean(f1_mac_avg), np.std(f1_mac_avg)
    print(f"\nAverage F1-macro score for lesion subtype classification across all folds: {f1_mac_mean:.3f} +/- {f1_mac_std:.3f}")

    df_folds.to_csv(os.path.join(args.path_model, 'metrics_all_subjects_all_folds.csv'), index=False)
    
    # average metrics across all folds
    df_mean = (df_folds.drop(columns=['reference', 'prediction', 'fold', 'EmptyRef', 'EmptyPred']).groupby('label').
               agg(['mean', 'std']).reset_index())
    print(df_mean)

    # Convert multi-index to flat index
    df_mean.columns = ['_'.join(col).strip() for col in df_mean.columns.values]
    # Rename column `label_` back to `label`
    df_mean.rename(columns={'label_': 'label'}, inplace=True)

    # Rename columns
    df.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)
    df_mean.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)

    # format output up to 3 decimal places
    df = df.round(3)
    df_mean = df_mean.round(3)

    # # print the mean metrics to the console
    # print('\nMean and standard deviation of metrics across all subjects:')
    # print(df_mean.to_string(index=False))
    # print(f'\nF1-macro score for classification task: {f1_macro_classification:.3f}')

    # # save as CSV
    # fname_output_csv = os.path.abspath(args.output)
    # df.to_csv(fname_output_csv, index=False)
    # print(f'Saved metrics to {fname_output_csv}.')

    # save as CSV
    fname_output_csv_mean = os.path.abspath(args.output.replace('.csv', '_mean.csv'))
    df_mean.to_csv(fname_output_csv_mean, index=False)
    print(f'Saved mean and standard deviation of metrics across all subjects to {fname_output_csv_mean}.')


if __name__ == '__main__':
    main()