import argparse
import os
import sys

# Get the absolute path of the parent directory
parent_dir = os.getcwd()

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import numpy as np
import optuna as op
import pytorch_lightning as pl
import torch

from dataset import Dataset
from model import QIBModel


def objective(trial):
    max_epochs = 5000
    batch_size = 512

    if args.optimize_hyperparameters:
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        n_layers = trial.suggest_int("n_layers", 1, 10)
        layers = []

        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 50, 400))

        latent_dim = trial.suggest_int("latent_dim", 16, 256)
        lr = trial.suggest_float("dropout_rate", 0.1, 1e-5)
        lr_decay = trial.suggest_float("dropout_rate", 0.999, 0.85)
    else:
        ae_layers = [512, 256, 128]
        latent_dim = 64
        mlp_layers = [512, 256, 128, 64]
        dropout_rate = 0.15
        lr = 1e-3
        lr_decay = 0.99

    # Data
    dataset = Dataset(args.dataset_path)
    train_split = 0.9

    # Get indices for train/test split
    indices = list(range(len(dataset)))
    split = int(np.floor(train_split * len(dataset)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Create train abset(dataset, train_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Dataset insights
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    in_dim, out_dim = dataset.get_dims()

    # Model architecture
    model = QIBModel(
        in_dim,
        ae_layers,
        latent_dim,
        args.positive_class,
        mlp_layers,
        dropout_rate=dropout_rate,
    )

    # # Weights Initialization
    # def weights_init(m):
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(
    #             m.weight, mode="fan_in", nonlinearity="leaky_relu"
    #         )
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)

    # model.apply(weights_init)

    # Train loop
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", patience=30, verbose=True, mode="min", min_delta=0.001
        ),
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=args.output_path,
            filename="{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=2,
        ),
        # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
    ]

    if args.optimize_hyperparameters:
        callbacks.append(
            op.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")
        )

    logger = pl.loggers.TensorBoardLogger(args.logs_path)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        benchmark=True,
        callbacks=callbacks,
        precision="64",
        profiler="simple",
        log_every_n_steps=2,
        logger=logger,
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    print("Done!")

    # Evaluate the model
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        required=False,
        default="../dataset/crop_bbox/",
    )
    parser.add_argument(
        "-o", "--output_path", required=False, default="../models/phenotype_classifier"
    )
    parser.add_argument(
        "-l", "--logs_path", required=False, default="../logs/phenotype_classifier"
    )
    parser.add_argument("-op", "--optimize_hyperparameters", action="store_true")
    parser.add_argument("-p", "--positive_class", required=False, type=int)
    args = parser.parse_args()

    # Seeds for reproducibility
    np.random.seed(123)
    torch.manual_seed(123)

    # Hyperparameter optimization
    if args.optimize_hyperparameters:
        study_name = "phenotype_classifier"
        storage_name = "sqlite:///{}.db".format(study_name)
        study = op.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(objective, n_trials=1000)
        print(study.best_params)
    else:
        objective(None)
