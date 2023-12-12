from typing import Any
import pytorch_lightning as pl
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layers=[512, 256, 128, 64], dropout_rate=0.1) -> None:
        super().__init__()
        layers_list = []
        prev_dim = input_dim

        for layer_dim in layers:
            layers_list.append(torch.nn.Linear(prev_dim, layer_dim))
            layers_list.append(torch.nn.Tanh())
            prev_dim = layer_dim

        layers_list.append(torch.nn.Dropout(dropout_rate))
        layers_list.append(torch.nn.Linear(prev_dim, 1))
        layers_list.append(torch.nn.Tanh())
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
        input_dim: (int, int, int) = (64, 128, 128),
        ae_hidden_dims: [int] = [512, 256],
        latent_dim: int = 128,
        positive_class: int = None,
        mlp_layers: [int] = None,
        dropout_rate: float = 0.5,
        fine_tuning: bool = False,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()

        # AutoEncoder
        self.ae = AutoEncoder(input_dim, ae_hidden_dims, latent_dim, dropout_rate)

        # MLP classifier for
        self.positive_class = positive_class
        if positive_class is not None:
            self.mlp = MLP(latent_dim, mlp_layers, dropout_rate)

        # Custom loss function for singular or bundle training
        if fine_tuning:
            self.loss_func = QIBCompositeLoss()  # TODO: for later
        else:
            self.loss_func = QIBLoss()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        X, Y = batch
        X_hat, z = self.ae(X)

        if self.positive_class is not None:
            score = self.mlp(z)
            loss = self.loss_func(X_hat, Y, score)
        else:
            loss = self.loss_func(X_hat, X)

        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        X_hat, z = self.ae(X)

        if self.positive_class is not None:
            score = self.mlp(z)
            loss = self.loss_func(score, Y, self.positive_class)
        else:
            loss = self.loss_func(X_hat, X)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def forward(self, x):
        # TODO: support hole pipeline when fine tuning
        if self.positive_class is None:
            (_, z) = self.ae(x)
            return z
        else:
            return self.mlp(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class QIBLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, positive_class=None):
        if positive_class is not None:
            # TODO: Missing class weights
            return torch.nn.functional.cross_entropy(pred, target)
        else:
            return torch.nn.functional.mse_loss(pred, target)


class QIBCompositeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target, score):
        pass
