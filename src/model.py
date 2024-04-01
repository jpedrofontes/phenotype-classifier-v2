import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix


class MLP(torch.nn.Module):
    
    def __init__(self, input_dim, layers=[512, 256, 128, 64], dropout_rate=0.1) -> None:
        super().__init__()
        layers_list = []
        prev_dim = input_dim

        for layer_dim in layers:
            layers_list.append(torch.nn.Linear(prev_dim, layer_dim))
            layers_list.append(torch.nn.ReLU())
            prev_dim = layer_dim

        layers_list.append(torch.nn.Dropout(dropout_rate))
        layers_list.append(torch.nn.Linear(prev_dim, 1))
        layers_list.append(torch.nn.Sigmoid())
        self.network = torch.nn.Sequential(*layers_list)

    def forward(self, x):
        return self.network(x)


class AutoEncoder(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate):
        super(AutoEncoder, self).__init__()
        self.hidden_dims = hidden_dims

        # Encoder
        modules = []
        in_channels = 1  # Assuming input has one channel; change if different
        for h_dim in hidden_dims:
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv3d(
                        in_channels, h_dim, kernel_size=3, stride=2, padding=1
                    ),
                    # torch.nn.BatchNorm3d(h_dim),
                    torch.nn.ReLU(),
                )
            )
            in_channels = h_dim
        modules.append(torch.nn.Dropout(dropout_rate))

        self.encoder = torch.nn.Sequential(*modules)

        # Determine the size of the encoded representation
        with torch.no_grad():
            self.sample_input = torch.zeros(1, *input_dim).unsqueeze(
                0
            )  # Create a sample input
            self.sample_encoded = self.encoder(self.sample_input)
            self.flattened_size = int(
                torch.prod(torch.tensor(self.sample_encoded.size()[1:]))
            )

        # Fully connected layers
        self.fc1 = torch.nn.Linear(self.flattened_size, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, self.flattened_size)

        # Decoder
        modules = []
        hidden_dims.reverse()  # Reverse the hidden dimensions for decoding

        for i in range(len(hidden_dims) - 1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,  # Adjust these parameters as needed
                        padding=1,
                        output_padding=1,
                    ),
                    # torch.nn.BatchNorm3d(hidden_dims[i + 1]),
                    torch.nn.ReLU(),
                )
            )

        # Assuming the input has 1 channel
        final_output_channels = 1
        modules.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(
                    hidden_dims[-1],
                    final_output_channels,  # Adjusted to match the input channels
                    kernel_size=3,
                    stride=2,  # Adjust these parameters as needed
                    padding=1,
                    output_padding=1,
                ),
                # torch.nn.BatchNorm3d(final_output_channels),
                torch.nn.ReLU(),
            )
        )
        modules.append(torch.nn.Dropout(dropout_rate))
        self.decoder = torch.nn.Sequential(*modules)

    def forward(self, X):
        if X.ndim == 4:
            X = X.unsqueeze(
                1
            ).contiguous()  # Add channel dimension and ensure it's contiguous

        # Encoding
        encoded = self.encoder(X)
        encoded_flat = encoded.view(encoded.size(0), -1).contiguous()
        z = self.fc1(encoded_flat)
        decoded_flat = self.fc2(z)
        decoded_flat = decoded_flat.view(
            decoded_flat.size(0), *self.sample_encoded.size()[1:]
        ).contiguous()
        X_hat = self.decoder(decoded_flat).squeeze(1).contiguous()

        return X_hat, z


class QIBModel(pl.LightningModule):

    def __init__(
        self,
        input_dim: tuple[int, int, int] = (64, 128, 128),
        ae_hidden_dims: list[int] = [16, 128, 32, 32],
        latent_dim: int = 256,
        positive_class: int = None,
        mlp_layers: list[int] = None,
        dropout_rate: float = 0.1,
        fine_tuning: bool = False,
        lr: float = 0.0007362490622651884,
        lr_decay: float = 0.9457240180432849,
        class_weights: dict = None,
    ):
        super().__init__()

        self.positive_class = positive_class
        self.class_weights = class_weights

        if positive_class is not None:
            self.mlp = MLP(latent_dim, mlp_layers, dropout_rate)
            self.acc_func = torchmetrics.classification.BinaryAccuracy()
            self.auc_func = torchmetrics.classification.BinaryAUROC(thresholds=None)
            self.precision_func = torchmetrics.classification.BinaryPrecision()
            self.recall_func = torchmetrics.classification.BinaryRecall()
            self.f1_score_func = torchmetrics.classification.BinaryF1Score()
        else:
            self.ae = AutoEncoder(input_dim, ae_hidden_dims, latent_dim, dropout_rate)

        # Custom loss function for singular or bundle training
        if fine_tuning:
            self.loss_func = QIBCompositeLoss()  # TODO: for later
        else:
            self.loss_func = QIBLoss(positive_class)

        self.save_hyperparameters()
        print(self.hparams)

    def training_step(self, batch, batch_idx):
        X, Y = batch

        if self.positive_class is not None:
            score = self.mlp(X)
            score = score.view(-1)
            Y = Y.view(-1)
            loss = self.loss_func(score, Y)
            acc = self.acc_func(score, Y)
            auc = self.auc_func(score, Y)
            precision = self.precision_func(score, Y)
            recall = self.recall_func(score, Y)
            f1_score = self.f1_score_func(score, Y)

            self.log(
                "acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "auc",
                auc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "precision",
                precision,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "recall",
                recall,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "f1_score",
                f1_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        else:
            X_hat, z = self.ae(X)
            weights = self.__compute_weights(Y)
            loss = self.loss_func(X_hat, X, weights)

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch

        if self.positive_class is not None:
            score = self.mlp(X)
            score = score.view(-1)
            Y = Y.view(-1)
            loss = self.loss_func(score, Y)
            acc = self.acc_func(score, Y)
            auc = self.auc_func(score, Y)
            precision = self.precision_func(score, Y)
            recall = self.recall_func(score, Y)
            f1_score = self.f1_score_func(score, Y)

            self.log(
                "val_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "val_auc",
                auc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "val_precision",
                precision,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "val_recall",
                recall,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "val_f1_score",
                f1_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        else:
            X_hat, z = self.ae(X)
            weights = self.__compute_weights(Y)
            loss = self.loss_func(X_hat, X, weights)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def forward(self, x):
        # TODO: support hole pipeline when fine tuning
        if self.positive_class is None:
            return self.ae(x)
        else:
            return self.mlp(x)

    def __compute_weights(self, labels):
        weights = torch.ones_like(labels, dtype=torch.float)
        for class_label, weight in self.class_weights.items():
            weights[labels == class_label] = weight
        return weights

    def evaluate_on_test_data(self, test_dataloader):
        self.mlp.eval()  # Set the model to evaluation mode

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data in test_dataloader:
                features, labels = data
                predictions = self.mlp(features)
                # Convert predictions to binary labels
                predictions = predictions.sigmoid() > 0.5  # Apply threshold if model outputs logits
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.int().cpu().numpy())

        # Build confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Optionally, convert to DataFrame for better readability
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual Class {i}" for i in range(len(cm))],
            columns=[f"Predicted Class {i}" for i in range(len(cm))],
        )

        return cm_df

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class QIBLoss(torch.nn.Module):
    
    def __init__(self, positive_class, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.positive_class = positive_class

        if self.positive_class is not None:
            self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target, weights):
        if self.positive_class is not None:
            return self.bce_loss(pred, target)
        else:
            # Weighted MSE Loss
            return torch.mean(weights * (pred - target) ** 2)


class QIBCompositeLoss(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target, score):
        pass
