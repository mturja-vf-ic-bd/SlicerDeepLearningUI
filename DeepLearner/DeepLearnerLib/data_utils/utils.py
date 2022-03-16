import os
import pandas as pd
import torch.utils.data
from monai.transforms import LoadImage, AddChannel, ScaleIntensity, EnsureType, Compose

from DeepLearnerLib.CONSTANTS import FILE_PATHS
from DeepLearnerLib.data_utils.CustomDataset import GeomCnnDataset


def get_image_files(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    attr = get_attributes()
    for sub in subject_ids:
        feat_tuple = []
        sub_path = os.path.join(FILE_PATHS[data_dir], sub)
        if not os.path.isdir(sub_path):
            continue
        n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                  if os.path.isdir(os.path.join(sub_path, f))]
        if len(n_feat) == 0:
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        group = sub_attr["group"].values[0]
        if group == "LR-neg":
            continue
        elif group == "HR-neg":
            labels.append(0)
        else:
            labels.append(1)
        for i, s in enumerate(scalars):
            feat_tuple.append(os.path.join(sub_path, s, "left_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
            feat_tuple.append(os.path.join(sub_path, s, "right_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
        file_names.append(feat_tuple)
    return file_names, labels


def get_image_files_2(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    attr = get_attributes()
    for sub in subject_ids:
        sub_path = os.path.join(FILE_PATHS[data_dir], sub)
        if not os.path.isdir(sub_path):
            continue
        n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                  if os.path.isdir(os.path.join(sub_path, f))]
        if len(n_feat) == 0:
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        for i, s in enumerate(scalars):
            feat_tuple = [os.path.join(sub_path, s, "left_" + s +
                                       FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg",
                          os.path.join(sub_path, s, "right_" + s +
                                       FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg"]
            file_names.append(feat_tuple)
            labels.append(i)
    return file_names, labels


def get_image_files_3(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    time_points = FILE_PATHS["TIME_POINTS"]
    attr = get_attributes()
    count = {"HR-neg": 0, "HR-ASD": 1}
    for sub in subject_ids:
        feat_tuple = []
        sub_path = os.path.join(FILE_PATHS[data_dir], sub, time_points[0])
        if not os.path.isdir(sub_path):
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
        for i, s in enumerate(scalars):
            feat_tuple.append(os.path.join(sub_path, s, "left_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
            feat_tuple.append(os.path.join(sub_path, s, "right_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
        file_names.append(feat_tuple)
    print(count)
    return file_names, labels


def get_test_dataloader():
    test_files, test_labels = get_image_files_3("TEST_DATA_DIR")
    test_transform = Compose(
        [LoadImage(image_only=True),
         AddChannel(),
         ScaleIntensity(),
         EnsureType()]
    )
    _ds = GeomCnnDataset(test_files, test_labels, test_transform)
    return torch.utils.data.DataLoader(_ds, batch_size=100)


def get_attributes():
    file_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "DX_and_Dem.csv")
    attr = pd.read_csv(open(file_path))
    return attr


def get_counts(data_dir):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"][0]
    time_points = FILE_PATHS["TIME_POINTS"]
    attr = get_attributes()

    for tp in ["V06", "V12", "V24"]:
        count = {"HR-neg": 0, "HR-ASD": 1}
        for sub in subject_ids:
            sub_path = os.path.join(FILE_PATHS[data_dir], sub, tp)
            if not os.path.isdir(sub_path):
                continue
            n_feat = os.path.join(sub_path, scalars,
                                  "left_" + scalars +
                                  FILE_PATHS["FILE_SUFFIX"][0] + ".jpeg")
            if not os.path.exists(n_feat):
                continue
            sub_attr = attr.loc[attr["CandID"] == int(sub)]
            if sub_attr.size == 0:
                continue
            group = sub_attr["group"].values[0]
            if group == "LR-neg":
                continue
            elif group == "HR-neg":
                labels.append(0)
            else:
                labels.append(1)
            count[group] += 1
        print(f"Time {tp}, Count {count}")


if __name__ == '__main__':
    get_counts("TEST_DATA_DIR")



