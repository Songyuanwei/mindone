import os

import PIL.Image as PImage

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def load_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        vision.Resize(mid_reso, interpolation=Inter.BICUBIC),
        # Resize: resize the shorter edge to mid_reso
        vision.RandomCrop((final_reso, final_reso)),
        vision.ToTensor(),
        normalize_01_into_pm1,
    ], [
        vision.Resize(mid_reso, interpolation=Inter.BICUBIC),
        # Resize: resize the shorter edge to mid_reso
        vision.CenterCrop((final_reso, final_reso)),
        vision.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, vision.RandomHorizontalFlip())
    train_aug, val_aug = vision.Compose(train_aug), vision.Compose(val_aug)

    # build dataset
    train_set = ds.ImageFolderDataset(
        dataset_dir=os.path.join(data_path, "train"), decode=True, extensions=[".JPEG", ".png"]
    )
    val_set = ds.ImageFolderDataset(
        dataset_dir=os.path.join(data_path, "val"), decode=True, extensions=[".JPEG", ".png"]
    )

    train_set = train_set.map(operations=train_aug, input_columns=["image"])
    val_set = val_set.map(operations=val_aug, input_columns=["image"])

    num_classes = 1000

    return num_classes, train_set, val_set
