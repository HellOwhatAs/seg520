import os
import polars
import numpy as np
import cv2
from numba import njit
from numba.typed.typedlist import List
from torch.utils.data import Dataset
import albumentations as A
from copy import deepcopy
import glob
import re


def extract_glob_stars(pattern: str, text: str):
    parts = pattern.split("*")
    escaped = list(map(re.escape, parts))
    regex = "^" + "(.*)".join(escaped) + "$"
    m = re.match(regex, text)
    return m.groups() if m else None


@njit
def _pixels2mask(mask: np.ndarray, pixels: List[int], class_id: int):
    fill_val = class_id
    for i in range(0, len(pixels), 2):
        mask[pixels[i] - 1 : pixels[i] - 1 + pixels[i + 1]] = fill_val


def pixels2mask(pixels: List[int], mask: np.ndarray, class_id: int = 1) -> np.ndarray:
    flattened_mask = mask.reshape(-1, order="C")
    _pixels2mask(flattened_mask, pixels, class_id)
    return flattened_mask.reshape(mask.shape, order="C")


@njit
def _mask2pixels(flattened_mask: np.ndarray, class_id: int) -> List[int]:
    fill_val = class_id
    pixels = List()
    idx = 0
    start_idx = 0
    count = 0
    for i in flattened_mask:
        if i == fill_val:
            if count == 0:
                start_idx = idx
            count += 1
        else:
            if count > 0:
                pixels.append(start_idx + 1)
                pixels.append(count)
                count = 0
        idx += 1
    if count > 0:
        pixels.append(start_idx + 1)
        pixels.append(count)
    return pixels


def mask2pixels(mask: np.ndarray, class_id: int = 1) -> List[int]:
    return _mask2pixels(mask.flatten("C"), class_id)


class UwDataset(Dataset):
    LABEL2ID = {
        "large_bowel": 0,
        "small_bowel": 1,
        "stomach": 2,
    }

    def __init__(
        self,
        images_pattern: str = "../input/train_images/",
        csv_path: str = "../input/train.csv",
        label2id: dict[str, int] = None,
        augmentation: A.BaseCompose = None,
    ):
        self.augmentation = augmentation
        self.label2id = label2id if label2id is not None else self.LABEL2ID
        self.images_pattern = images_pattern.replace("\\", "/")
        self.csv_path = csv_path
        self.image_paths = [
            os.path.abspath(i).replace("\\", "/")
            for i in glob.glob(self.images_pattern)
        ]
        self.df = polars.read_csv(self.csv_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        (_, case, day, slice, w, h, _, _) = extract_glob_stars(
            self.images_pattern, image_path
        )
        height, width = image.shape
        assert height == int(h) and width == int(w)
        image_id = f"case{case}_day{day}_slice_{slice}"

        mask: np.ndarray = np.zeros(
            (height, width, len(self.label2id)), dtype=image.dtype
        )

        for _, class_id, encoded_pixels in self.df.filter(
            polars.col("id") == image_id
        ).iter_rows():
            if encoded_pixels is None:
                continue
            mask[:, :, self.label2id[class_id]] = pixels2mask(
                List(map(int, str.split(encoded_pixels))),
                mask[:, :, self.label2id[class_id]],
            )
            assert (
                " ".join(map(str, mask2pixels(mask[:, :, self.label2id[class_id]])))
                == encoded_pixels
            )

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return np.expand_dims(image, axis=0), mask.transpose(2, 0, 1)

    def subset(self, indices: list[int]):
        subset = deepcopy(self)
        subset.image_paths = [self.image_paths[i] for i in indices]
        return subset


def get_train_augmentation(height: int = 384, width: int = 384):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.OneOf(
            [
                A.Compose(
                    [
                        A.PadIfNeeded(
                            min_height=height, min_width=width, always_apply=True
                        ),
                        A.RandomCrop(height=height, width=width, always_apply=True),
                    ]
                ),
                A.Resize(height=height, width=width),
            ],
            p=1,
        ),
    ]
    return A.Compose(test_transform)


def get_validation_augmentation(height: int = 384, width: int = 384):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height=height, width=width),
    ]
    return A.Compose(test_transform)


if __name__ == "__main__":
    dataset = UwDataset(
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train/case*/case*_day*/scans/slice_*_*_*_*_*.png",
        "D:/Downloads/uw-madison-gi-tract-image-segmentation/train.csv",
        augmentation=get_train_augmentation(),
    )

    for fpath, (img, mask) in zip(dataset.image_paths, dataset):
        print(fpath)
        cv2.imshow(
            "img",
            cv2.addWeighted(
                np.permute_dims(mask * 255, (1, 2, 0)),
                0.3,
                cv2.cvtColor(cv2.equalizeHist(img[0]), cv2.COLOR_GRAY2BGR),
                0.7,
                0,
            ),
        )
        cv2.waitKey(1)
