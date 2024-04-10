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

from dataset import BreastCancerDataset, phenotypes
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
            mlp_layers = [512, 256, 64]
        else:
            mlp_layers = []
        dropout_rate = 0.0
        lr = 0.001
        lr_decay = 0.99

    in_dim, out_dim = dataset.get_dims()

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
        class_weights=dataset.class_weights,
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
        # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
    ]

    if args.optimize_hyperparameters:
        callbacks.append(
            op.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")
        )
    else:
        callbacks.append(
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=args.output_path,
                filename="{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=2,
            ),
        )

    logger = pl.loggers.TensorBoardLogger(args.logs_path)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        # precision="16-mixed",
        log_every_n_steps=1,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
    print("Done!")

    if args.positive_class is not None:
        confusion_matrix_df = model.evaluate_on_test_data(dataset.val_dataloader())
        print(confusion_matrix_df)

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
    batch_size = 512

    # Initialize DataModule
    dataset = BreastCancerDataset(
        base_path=args.dataset_path,
        positive_class=args.positive_class,
        batch_size=batch_size,
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
