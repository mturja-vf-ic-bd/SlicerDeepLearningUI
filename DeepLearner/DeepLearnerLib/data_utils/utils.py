import os

import numpy as np
import pandas as pd
import torch.utils.data
from monai.transforms import LoadImage, AddChannel, ScaleIntensity, EnsureType, Compose

from DeepLearnerLib.CONSTANTS import DEFAULT_FILE_PATHS
from DeepLearnerLib.data_utils.CustomDataset import GeomCnnDataset


def get_image_files_single_scalar(data_dir="TRAIN_DATA_DIR", FILE_PATHS=None):
    file_names = []
    labels = []
    if FILE_PATHS is None:
        FILE_PATHS = DEFAULT_FILE_PATHS
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"][0]
    time_points = FILE_PATHS["TIME_POINTS"][0]
    attr = get_attributes(FILE_PATHS)
    count = {"HR-neg": 0, "HR-ASD": 1}
    for sub in subject_ids:
        feat_tuple = []
        sub_path = os.path.join(FILE_PATHS[data_dir], sub, time_points)
        if not os.path.isdir(sub_path) or not os.path.isdir(os.path.join(sub_path, scalars)):
            continue
        n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                  if os.path.isdir(os.path.join(sub_path, f))]
        if len(n_feat) == 0:
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        group = sub_attr["group"].values[0]
        if "LR" in group:
            continue
        elif group == "HR-neg":
            labels.append(0)
        else:
            labels.append(1)
        count[group] += 1
        feat_tuple.append(os.path.join(sub_path, scalars, "left_" + scalars +
                                       FILE_PATHS["FILE_SUFFIX"][0]) + ".jpeg")
        feat_tuple.append(os.path.join(sub_path, scalars, "right_" + scalars +
                                       FILE_PATHS["FILE_SUFFIX"][0]) + ".jpeg")
        file_names.append(feat_tuple)
    print(count)
    return file_names, labels


def get_test_dataloader():
    test_files, test_labels = get_image_files_single_scalar("TEST_DATA_DIR")
    test_transform = Compose(
        [LoadImage(image_only=True),
         AddChannel(),
         ScaleIntensity(),
         EnsureType()]
    )
    _ds = GeomCnnDataset(test_files, test_labels, test_transform)
    return torch.utils.data.DataLoader(_ds, batch_size=100)


def get_attributes(FILE_PATHS):
    file_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "DX_and_Dem.csv")
    attr = pd.read_csv(open(file_path))
    return attr


def anonymize_dataset(FILE_PATHS):
    import shutil
    path = "/Users/mturja/Desktop/sample_dataset"
    output_path = "/Users/mturja/Desktop/sample_dataset_anonymous"
    FILE_PATHS["TRAIN_DATA_DIR"] = path
    attr = get_attributes(FILE_PATHS)
    attr.drop(columns=["ASD_DX", "Gender", "Project", "HRLR", "MRI_Age", "Visit"], inplace=True)
    subject_ids = sorted(os.listdir(FILE_PATHS["TRAIN_DATA_DIR"]))
    subject_ids.remove(".DS_Store")
    subject_ids.remove("DX_and_Dem.csv")
    attr.drop(attr[~attr["CandID"].isin([int(s) for s in subject_ids])].index, inplace=True)
    for i, sub in enumerate(subject_ids):
        sub_path = os.path.join(path, sub)
        if not os.path.isdir(sub_path):
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        group = sub_attr["group"].values[0]
        if "LR" in group:
            attr.drop(attr[attr["CandID"] == int(sub)].index, inplace=True)
            continue
        print("Copying ... ")
        attr.loc[attr["CandID"] == int(sub), ["CandID", "group"]] = [i, group]
        shutil.copytree(os.path.join(path, sub), os.path.join(output_path, str(i)))
    attr.to_csv(os.path.join(output_path, "DX_and_Dem.csv"))


if __name__ == '__main__':
    anonymize_dataset(DEFAULT_FILE_PATHS)





