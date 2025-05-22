from uw_dataset import UwDataset, get_train_augmentation, get_validation_augmentation
from seg_model import SegModel
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="FPN", help="arch of seg model")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="max epochs")
    args = parser.parse_args()
    arch: str = getattr(args, "arch")
    encoder_name: str = getattr(args, "encoder_name")
    batch_size: int = getattr(args, "batch_size")
    epoch: int = getattr(args, "epoch")
    return {
        "arch": arch,
        "encoder_name": encoder_name,
        "batch_size": batch_size,
        "epoch": epoch,
    }


def main():
    FOLD = 0
    args = get_args()
    batch_size = args["batch_size"]
    epoch = args["epoch"]
    arch = args["arch"]
    encoder_name = args["encoder_name"]

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
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = SegModel(
        arch=arch,
        encoder_name=encoder_name,
        in_channels=1,
        out_classes=4,
        t_max=epoch * len(train_dataloader),
    )

    trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks=[
            ModelCheckpoint(monitor="valid_dataset_iou", mode="max"),
            EarlyStopping(monitor="valid_dataset_iou", mode="max", patience=20),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()
