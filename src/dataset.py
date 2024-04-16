import copy
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import NearMiss
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split

phenotypes = {
    0: "Luminal A",
    1: "Luminal B",
    2: "HER2-enriched",
    3: "Triple Negative",
}


class BreastCancerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        base_path: str,
        mode="train",
        positive_class: int = None,
        crop_size=(64, 128, 128),
        batch_size: int = 512,
    ):
        self.base_path = base_path

        if mode not in ["train", "test"]:
            raise ValueError("Invalid mode set for Dataset")

        self.mode = mode
        self.crop_size = crop_size
        self.positive_class = positive_class
        self.batch_size = batch_size
        self.volumes = {}
        self.discarded_volumes = {}

        if self.positive_class is None:
            self.__read_all_volumes()
            self.__prepare_and_balance_ae_dataset()
            self.class_counts = {}

            for idx in self.training_volumes:
                y = self.volumes[idx]["phenotype"]

                if y not in self.class_counts:
                    self.class_counts[y] = 1
                else:
                    self.class_counts[y] += 1

            total_samples = sum(self.class_counts.values())
            self.class_weights = {
                class_label: total_samples / count
                for class_label, count in self.class_counts.items()
            }

            # Normalize the weights
            max_weight = max(self.class_weights.values())
            self.class_weights = {
                k: v / max_weight for k, v in self.class_weights.items()
            }
        else:
            self.__read_dataset_for_mlp()

    def __read_all_volumes(self):
        classes_csv = pd.read_csv(os.path.join(self.base_path, "classes.csv"))

        for file_path in glob.glob(os.path.join(self.base_path, "*.jpg")):
            file = os.path.basename(file_path)

            # Get the case number of the image
            match_case = re.search("Breast_MRI_[0-9]+", file)
            case = file[match_case.start() : match_case.end()]

            # Get the series of the image
            match_series = re.search("_series#.+#_", file)
            series = file[match_series.start() : match_series.end()]

            if case is None or series is None:
                continue

            volume_name = f"{case}.{series}"

            # Match the case number with the phenotype class
            phenotype_row = classes_csv.loc[classes_csv['patient_id'] == case, 'mol_subtype']

            if not phenotype_row.empty:
                phenotype = phenotype_row.iloc[0]  
            else:
                continue

            self.volumes.setdefault(
                volume_name, {"case": case, "slices": [], "phenotype": phenotype}
            )["slices"].append(file_path)

    def __prepare_and_balance_ae_dataset(self):
        all_volume_keys = list(self.volumes.keys())
        train_keys, test_keys = train_test_split(
            all_volume_keys, test_size=0.1, random_state=42
        )

        # Balance the training dataset
        train_volumes = {k: self.volumes[k] for k in train_keys}
        balanced_train_volumes, discarded_volumes = self.__balance_ae_dataset(
            train_volumes
        )

        # Update training_volumes to only include keys of balanced volumes
        self.training_volumes = list(balanced_train_volumes.keys())

        # Combine original test keys with keys from discarded volumes for testing
        self.testing_volumes = test_keys + list(discarded_volumes.keys())

    def __balance_ae_dataset(self, volumes):
        # Count phenotypes
        phenotype_counts = {}
        for key, volume_data in volumes.items():
            phenotype = volume_data["phenotype"]
            if phenotype not in phenotype_counts:
                phenotype_counts[phenotype] = []
            phenotype_counts[phenotype].append(key)

        # Find the target balance count, which is the count of the second smallest class to avoid "discarding" too much data
        target_counts = sorted([len(keys) for keys in phenotype_counts.values()])
        if len(target_counts) > 1:
            target_count = target_counts[1]
        else:
            return volumes, {}  # If not enough classes to balance, return as is

        balanced_volumes = {}
        discarded_volumes = {}

        for phenotype, keys in phenotype_counts.items():
            if len(keys) > target_count:
                selected_keys = np.random.choice(keys, size=target_count, replace=False)
                for key in selected_keys:
                    balanced_volumes[key] = volumes[key]
                for key in set(keys) - set(selected_keys):
                    discarded_volumes[key] = volumes[key]
            else:
                for key in keys:
                    balanced_volumes[key] = volumes[key]

        return balanced_volumes, discarded_volumes

    def __read_dataset_for_mlp(self):
        # Read dataset
        df = pd.read_csv(os.path.join(self.base_path, "z_data.csv"), header=None)
        self.z_data = df
        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        # Edit phenotype according to positive class
        y = y.apply(lambda label: 1 if label == self.positive_class else 0)

        # Split train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=123
        )

        # Apply the undersampling transformation to the dataset
        undersampler = NearMiss(version=1, n_neighbors=3)
        self.X_train, self.y_train = undersampler.fit_resample(X_train, y_train)
        # TODO: add discarded samples to test
        self.X_test, self.y_test = X_test, y_test

        # Calculate class counts
        self.train_counts = {
            "0": len(self.y_train[self.y_train == 0]),
            "1": len(self.y_train[self.y_train == 1]),
        }
        self.test_counts = {
            "0": len(self.y_test[self.y_test == 0]),
            "1": len(self.y_test[self.y_test == 1]),
        }

    def set_mode(self, mode: str):
        self.mode = mode

    def __len__(self):
        if self.positive_class is None:
            if self.mode == "train":
                size = len(self.training_volumes)
            elif self.mode == "test":
                size = len(self.testing_volumes)
        else:
            if self.mode == "train":
                size = len(self.X_train)
            elif self.mode == "test":
                size = len(self.X_test)

        return size

    def __getitem__(self, idx: int):
        if self.positive_class is None:
            # Determine the correct set of volume keys to use based on mode
            volume_keys = (
                self.training_volumes if self.mode == "train" else self.testing_volumes
            )
            volume_key = volume_keys[idx]

            # Process the scan and fetch the label
            X = self.process_scan(self.volumes[volume_key]["slices"])
            y = self.volumes[volume_key]["phenotype"]
        else:
            if self.mode == "train":
                X = self.X_train.iloc[idx, :]
                y = self.y_train.iloc[idx]
            else:
                X = self.X_test.iloc[idx, :]
                y = self.y_test.iloc[idx]

            # Convert to appropriate tensor format
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def read_volume(self, volume_name: str):
        volume = []

        for img_path in self.volumes[volume_name]["slices"]:
            # Read the image
            img = Image.open(img_path).convert("L")  # 'L' mode for grayscale
            img = img.resize((self.crop_size[0], self.crop_size[1]))
            img = np.array(img)
            # Add to volume
            volume.append(img)
        volume = np.array(volume)

        return volume

    def normalize(self, volume: np.ndarray):
        """Normalize the volume"""
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def resize_volume(
        self,
        volume: np.ndarray,
        desired_width: int = 128,
        desired_height: int = 128,
        desired_depth: int = 64,
    ):
        """Resize across z-axis"""
        # Get current depth
        current_depth = volume.shape[0]
        current_width = volume.shape[1]
        current_height = volume.shape[2]

        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height

        # Resize across z-axis
        volume = ndimage.zoom(
            volume, (depth_factor, width_factor, height_factor), order=1
        )

        return volume

    def process_scan(self, key: str):
        """Read and resize volume"""
        # Read scan
        volume = self.read_volume(key)

        # Normalize
        volume = self.normalize(volume)

        # Resize width, height and depth
        volume = self.resize_volume(
            volume,
            desired_depth=self.crop_size[0],
            desired_width=self.crop_size[1],
            desired_height=self.crop_size[2],
        )

        return volume

    def get_dims(self):
        if self.positive_class is None:
            return (self.crop_size, 1)
        else:
            return (len(self.z_data.columns) - 1, 1)

    def get_class_sample_counts(self):
        if self.positive_class is not None:
            if self.mode == "train":
                return self.train_counts
            elif self.mode == "test":
                return self.test_counts
        else:
            raise ValueError(
                "class_sample_counts is only available when positive_class is not None"
            )

    def train_dataloader(self):
        # Create and return a DataLoader for the training dataset
        dataset = copy.deepcopy(self)
        dataset.set_mode("train")

        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        # Create and return a DataLoader for the testing dataset
        dataset = copy.deepcopy(self)
        dataset.set_mode("test")

        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    import csv
    import sys
    import torch

    from model import QIBModel

    # Best params for auto encoder
    # TODO: Read from json config
    model_params = {
        "ae_hidden_dims": [32, 32, 128, 16],
        "dropout_rate": 0.1,
        "fine_tuning": False,
        "input_dim": (64, 128, 128),
        "latent_dim": 256,
        "lr": 0.0007362490622651884,
        "lr_decay": 0.9457240180432849,
        "mlp_layers": [],
        "positive_class": None,
    }

    # Load the MRI dataset
    dataset = BreastCancerDataset(sys.argv[1])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=128
    )

    # Load auto encoder model from checkpoint
    model = QIBModel.load_from_checkpoint(sys.argv[2], **model_params)
    model = model.to("cuda")
    model.eval()

    # Get Z from auto encoder and associate with the phenotype
    z_data = []

    for x, y in iter(dataloader):
        x_hat, z = model(x.cuda())

        # Flatten z if it's multidimensional and convert to list
        z_flat = z.view(-1).tolist()

        # Append label y to the flattened z list
        z_data.append(z_flat + [int(y)])

    with open(sys.argv[3], mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in z_data:
            writer.writerow(row)

    print(f"z_data exported to {sys.argv[3]}")
