from uw_dataset import UwDataset, get_train_augmentation, get_validation_augmentation
from seg_model import MultiLabelSegmentModule
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader


def main():
    BATCH_SIZE = 32
    EPOCHS = 50
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
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiLabelSegmentModule(
        arch="FPN",
        encoder_name="resnet34",
        in_channels=1,
        out_classes=3,
        t_max=EPOCHS * len(train_dataloader),
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="valid_diceloss", mode="min"),
            EarlyStopping(monitor="valid_diceloss", mode="min", patience=20),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()
