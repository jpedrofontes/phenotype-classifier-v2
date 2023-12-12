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
    import csv
    import sys
    import torch

    from model import QIBModel

    # Best params for auto encoder
    # TODO: Read from json config
    model_params = {
        "input_dim": (64, 128, 128),
        "ae_hidden_dims": [64, 32],
        "latent_dim": 128,
        "positive_class": None,
        "mlp_layers": [],
        "dropout_rate": 0.1,
        "fine_tuning": False,
        "lr": 1e-6,
        "lr_decay": 0.96,
    }

    # Load the MRI dataset
    dataset = Dataset(sys.argv[1])
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
