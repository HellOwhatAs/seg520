from uw_dataset import UwDataset, get_validation_augmentation
from seg_model import SegModel
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 100
    FOLD = 0

    dataset = UwDataset(
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train/case*/case*_day*/scans/slice_*_*_*_*_*.png",
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train.csv",
        augmentation=get_validation_augmentation(),
    )

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    folds = list(kf.split(list(range(len(dataset)))))
    train_index, valid_index = folds[FOLD]
    train_dataset, valid_dataset = (
        dataset.subset(train_index),
        dataset.subset(valid_index),
    )
    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegModel.load_from_checkpoint(
        checkpoint_path="lightning_logs/without_a/checkpoints/epoch=95-step=92448.ckpt",
        map_location=DEVICE,
        arch="FPN",
        encoder_name="resnet34",
        in_channels=1,
        out_classes=4,
        t_max=EPOCHS * len(train_dataloader),
    ).eval()

    with torch.no_grad():
        for img, mask in valid_dataset:
            pred: torch.Tensor = model(
                torch.tensor(img, dtype=torch.float, device=model.device).unsqueeze(0)
            )
            pred = pred.contiguous().softmax(dim=1).argmax(dim=1).squeeze()
            pred = pred.cpu().numpy().astype(np.uint8)

            pred_bgr: np.ndarray = cv2.merge(
                [np.astype(pred == (i + 1), np.uint8) for i in range(3)]
            )
            mask_bgr: np.ndarray = cv2.merge(
                [np.astype(mask == (i + 1), np.uint8) for i in range(3)]
            )

            img: np.ndarray = img[0]
            img_bgr: np.ndarray = cv2.cvtColor(
                cv2.equalizeHist(img)
                if np.unique(img).shape[0] > 10
                else img * (255 // (img.max() + 1)),
                cv2.COLOR_GRAY2BGR,
            )

            cv2.imshow("mask", cv2.addWeighted(mask_bgr * 255, 0.4, img_bgr, 0.6, 0))
            cv2.imshow("pred", cv2.addWeighted(pred_bgr * 255, 0.4, img_bgr, 0.6, 0))
            cv2.waitKey(0)
