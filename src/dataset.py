import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch

phenotypes = {
    0: "Luminal A",
    1: "Luminal B",
    2: "HER2-enriched",
    3: "Triple Negative",
}


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, base_path: str, crop_size=(64, 128, 128), positive_class: int = 0
    ):
        super().__init__()
        self.base_path = base_path
        self.crop_size = crop_size
        self.volumes = dict()

        # Read class csv file
        classes_csv = pd.read_csv(os.path.join(self.base_path, "classes.csv"))

        # Build dataset info
        os.chdir(base_path)
        print("Reading dataset...")

        for file in glob.glob("*.jpg"):
            # Get the case number of the image
            match_case = re.search("Breast_MRI_[0-9]+", file)
            case = file[match_case.start() : match_case.end()]

            if case is None:
                print("will continue case none")
                continue

            # Get the series of the image
            match_series = re.search("_series#.+#_", file)
            series = file[match_series.start() : match_series.end()]

            if series is None:
                print("will continue series none")
                continue

            # Save volume info
            volume_name = "{}.{}".format(case, series)

            # Match the case number with the phenotype class
            phenotype = classes_csv.loc[classes_csv["patient_id"] == case][
                "mol_subtype"
            ]

            if phenotype is None:
                continue
            else:
                phenotype = phenotype.tolist()[0]
                try:
                    self.volumes[volume_name]["case"] = case
                    self.volumes[volume_name]["slices"].append(
                        os.path.join(base_path, file)
                    )
                    self.volumes[volume_name]["slices"].sort()
                    self.volumes[volume_name]["phenotype"] = phenotype
                except:
                    self.volumes[volume_name] = dict()
                    self.volumes[volume_name]["case"] = case
                    self.volumes[volume_name]["slices"] = [
                        os.path.join(base_path, file)
                    ]
                    self.volumes[volume_name]["phenotype"] = phenotype

        self.volumes_idx = list(self.volumes.keys())
        self.positive_class = positive_class

    def __len__(self):
        return len(self.volumes_idx)

    def __getitem__(self, idx: int):
        X = self.process_scan(self.volumes_idx[idx])

        # Consider one phenotype as 1 and the other as 0
        phenotype = self.volumes[self.volumes_idx[idx]]["phenotype"]
        y = 1 if phenotype == self.positive_class else 0

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
        return (self.crop_size, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Seeds for reproducibility
    np.random.seed(123)

    BASE_PATH = "/home/jpedrofontes/phenotype-classifier-v2/"
    BASE_DATA_PATH = "/home/jpedrofontes/phenotype-classifier-v2/dataset/"

    dataset = Dataset(f"{BASE_DATA_PATH}/crop_bbox", crop_size=(64, 128, 128))
    print(dataset.get_dims())

    print("\nCases/Series: %d" % len(dataset.volumes_idx))
    key = dataset.volumes_idx[0]
    print("\nExample:\n\nCase/Series: %s" % key)
    print(
        "Phenotype: %s => %s"
        % (
            dataset.volumes[key]["phenotype"],
            phenotypes[dataset.volumes[key]["phenotype"]],
        )
    )
    print("Number of tumor slices: %d" % len(dataset.volumes[key]["slices"]))
    volume, _ = dataset[0]
    print(f"\nVolume shape (after pre-processing): {volume.shape}")

    fig = plt.figure()
    fig.suptitle(
        "Tumor Phenotype: {}".format(phenotypes[dataset.volumes[key]["phenotype"]])
    )
    ax = plt.subplot(2, 3, 1)
    ax.title.set_text("Slice 1")
    plt.imshow(volume[0, :, :].T)
    ax = plt.subplot(2, 3, 2)
    ax.title.set_text("Slice 20")
    plt.imshow(volume[19, :, :].T)
    ax = plt.subplot(2, 3, 3)
    ax.title.set_text("Slice 30")
    plt.imshow(volume[29, :].T)
    ax = plt.subplot(2, 3, 4)
    ax.title.set_text("Slice 40")
    plt.imshow(volume[39, :, :].T)
    ax = plt.subplot(2, 3, 5)
    ax.title.set_text("Slice 50")
    plt.imshow(volume[49, :, :].T)
    ax = plt.subplot(2, 3, 6)
    ax.title.set_text("Slice 64")
    plt.imshow(volume[63, :, :].T)
    plt.show()
