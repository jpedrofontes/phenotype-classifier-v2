import argparse
import gc
import os
import sys
import uuid

# Get the absolute path of the parent directory
parent_dir = os.getcwd()

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import numpy as np
import optuna as op
import lightning.pytorch as pl
import torch

torch.cuda.empty_cache()
torch.set_float32_matmul_precision("medium")

from dataset import Dataset, phenotypes
from model import QIBModel


def objective(trial):
    max_epochs = 5000

    if args.optimize_hyperparameters:
        if args.positive_class is not None:
            n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 10)
            mlp_layers = []

            for i in range(n_mlp_layers):
                mlp_layers.append(
                    trial.suggest_categorical(
                        f"mlp_units_l{i}", [256, 512, 1024, 2048, 4096]
                    )
                )
            latent_dim = 256
            ae_layers = [32, 32, 128, 16]
        else:
            n_ae_layers = trial.suggest_int("n_layers", 1, 5)
            ae_layers = []

            for i in range(n_ae_layers):
                ae_layers.append(
                    trial.suggest_categorical(f"ae_units_l{i}", [16, 32, 64, 128, 256])
                )
            latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64, 128, 256])
            mlp_layers = []

        dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])
        lr = trial.suggest_float("lr", 1e-6, 0.1)
        lr_decay = trial.suggest_float("lr_decay", 0.9, 0.99)
    else:
        ae_layers = [32, 32, 128, 16]
        latent_dim = 256
        if args.positive_class is not None:
            mlp_layers = [16, 32, 128, 32, 16, 64, 64, 32, 128, 64]
        else:
            mlp_layers = []
        dropout_rate = 0.1
        lr = 0.020037704053394004  # 0.0007362490622651884
        lr_decay = 0.9558576064076237  # 0.9457240180432849

    # Dataset insights
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    in_dim, out_dim = dataset.get_dims()

    if args.positive_class is not None:
        class_counts = dataset.get_class_sample_counts()
    else:
        class_counts = None

    # Model architecture
    model = QIBModel(
        in_dim,
        ae_layers,
        latent_dim,
        args.positive_class,
        mlp_layers,
        dropout_rate=dropout_rate,
        # fine_tuning = False,
        lr=lr,
        lr_decay=lr_decay,
        class_counts=class_counts,
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
            monitor="val_loss", patience=10, verbose=True, mode="min", min_delta=0.001
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
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=1,
        logger=logger,
        enable_progress_bar=False,
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

    # Data
    batch_size = 128
    dataset = Dataset(args.dataset_path, positive_class=args.positive_class)
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    if args.positive_class is not None:
        storage_name = "sqlite:///studies/phenotypes.db"
        study_name = f"phenotype_classifier_{phenotypes[args.positive_class]}"
    else:
        storage_name = "sqlite:///studies/autoencoder.db"
        study_name = "autoencoder"

    # Hyperparameter optimization
    if args.optimize_hyperparameters:
        study = op.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(objective, n_trials=1000, gc_after_trial=True)
        print(study.best_params)
    else:
        objective(None)
