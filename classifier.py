from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._utils import patch_first_conv
from uw_dataset import UwDataset, get_train_augmentation, get_validation_augmentation
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import timm
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


class Classifier(pl.LightningModule):
    def __init__(self, model_name="resnet34", in_channels=1, num_classes=2):
        super(Classifier, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
        patch_first_conv(self.model, new_in_channels=in_channels, pretrained=True)
        params = smp.encoders.get_preprocessing_params(model_name)
        self.std: torch.Tensor
        self.register_buffer("std", torch.tensor(params["std"]).mean())
        self.mean: torch.Tensor
        self.register_buffer("mean", torch.tensor(params["mean"]).mean())

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        image = torch.tensor((image - self.mean) / self.std, dtype=torch.float)
        return self.model(image)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, F.one_hot(y, num_classes=2).float())
        acc = (y == y_hat.argmax(dim=1)).float().mean()
        return {
            "loss": loss,
            "acc": acc,
        }

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        return output["loss"]

    def shared_epoch_end(self, outputs, stage):
        metrics = {
            f"{stage}_loss": torch.stack([i["loss"] for i in outputs]).mean().item(),
            f"{stage}_acc": torch.stack([i["acc"] for i in outputs]).mean().item(),
        }
        self.log_dict(metrics, prog_bar=True)

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    BATCH_SIZE = 32
    EPOCHS = 100
    FOLD = 0
    dataset = UwDataset(
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train/case*/case*_day*/scans/slice_*_*_*_*_*.png",
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train.csv",
    )

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    folds = list(kf.split(list(range(len(dataset)))))
    train_index, valid_index = folds[FOLD]
    train_dataset, valid_dataset = (
        dataset.subset(train_index),
        dataset.subset(valid_index),
    )
    train_dataset.augmentation = get_train_augmentation()
    valid_dataset.augmentation = get_validation_augmentation()
    train_dataset.classify = True
    valid_dataset.classify = True

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Classifier("resnet34")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="valid_acc", mode="max"),
            EarlyStopping(monitor="valid_acc", mode="max", patience=20),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()
