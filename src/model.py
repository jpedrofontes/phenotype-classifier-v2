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
    def __init__(
        self,
        input_dim: (int, int, int),
        hidden_dims: [int],
        latent_dim: int,
        dropout_rate: float,
    ):
        super(AutoEncoder, self).__init__()
        self.hidden_dims = hidden_dims

        depth, height, width = input_dim
        channels = 1

        # Calculate resize factors
        self.depth_factor = depth // (2 ** len(hidden_dims)) ** 2
        self.height_factor = height // (2 ** len(hidden_dims)) ** 2
        self.width_factor = width // (2 ** len(hidden_dims)) ** 2

        # Encoder
        encoder_layers = []
        prev_channels = channels

        for next_channels in hidden_dims:
            encoder_layers.append(
                torch.nn.Conv3d(
                    prev_channels, next_channels, kernel_size=3, stride=2, padding=1
                )
            )
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.MaxPool3d(2))
            prev_channels = next_channels

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Calculate the flattened size of the encoder output
        self.flattened_size = (
            prev_channels * self.depth_factor * self.height_factor * self.width_factor
        )
        self.fc1 = torch.nn.Linear(self.flattened_size, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, self.flattened_size)

        # Decoder
        decoder_layers = []

        for next_channels in reversed(hidden_dims[:-1]):
            decoder_layers.append(torch.nn.Upsample(scale_factor=2, mode="trilinear"))
            decoder_layers.append(
                torch.nn.ConvTranspose3d(
                    prev_channels, next_channels, kernel_size=3, stride=1, padding=1
                )
            )
            decoder_layers.append(torch.nn.ReLU())
            prev_channels = next_channels

        decoder_layers.append(torch.nn.Upsample(scale_factor=2, mode="trilinear"))
        decoder_layers.append(
            torch.nn.ConvTranspose3d(
                prev_channels, channels, kernel_size=3, stride=1, padding=1
            )
        )
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        tmp = self.encoder(x)
        tmp = tmp.view(-1, self.flattened_size)  # Flatten op
        z = self.fc1(tmp)  # Latent representation
        tmp_hat = self.fc2(z)
        tmp_hat = tmp_hat.view(
            tmp_hat.size(0),
            self.hidden_dims[-1],
            self.depth_factor,
            self.height_factor,
            self.width_factor,
        )  # Reverse flatten
        x_hat = self.decoder(tmp_hat)
        return x_hat, z


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
        if positive_class is not None:
            self.positive_class = positive_class
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
            return self.loss_func(X_hat, Y, score)
        else:
            return self.loss_func(X_hat, X)

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        X_hat, z = self.ae(X)

        if self.positive_class is not None:
            score = self.mlp(z)
            return self.loss_func(score, Y, self.positive_class)
        else:
            return self.loss_func(X_hat, X)
        
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
    def __init__(self, alpha: float=1.0) -> None:
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
