import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torchvision.models.vision_transformer as vit
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt
from torchinfo import summary
import matplotlib.pyplot as plt

from typing import Tuple, Never


def set_device() -> str:
    """
    使用するデバイスを指定する
    """
    # デバイスの指定
    device = (
        "cuda" 
        if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device\n")
    return device


def fix_seed(device: str, seed: int = 0):
    """
    各種乱数シードの固定
    """
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)


def confirm_dataset(train_data: datasets, val_data: datasets, test_data: datasets) -> None:
    """
    学習データ、検証データ、テストデータのサイズ確認
    """
    print(f"\nDataset: {train_data.__class__.__name__}")
    print(f"    Training data  : {len(train_data)}")
    print(f"    Validation data: {len(val_data)}")
    print(f"    Test data      : {len(test_data)}\n")


def get_img_info(dataloader: DataLoader) -> Tuple[int, int] | Never:
    """
    データローダーの形状を表示する。返り値としてチャンネル数と画像サイズを返す。
    """
    class NotSquareImgError(Exception):
        def __str__(self):
            return f"{NotSquareImgError.__name__}: Image height and width don't match!"

    for X, y in dataloader:
        print("Shape of X")
        print(f"    Batch size: {X.shape[0]}")
        print(f"    Channels  : {X.shape[1]}")
        print(f"    Height    : {X.shape[2]}")
        print(f"    Width     : {X.shape[3]}")
        print(f"Shape of y : {y.shape} {y.dtype}\n")
        channels = X.shape[1]
        img_size = X.shape[2]
        break

    try:
        if X.shape[2] != X.shape[3]:
            raise NotSquareImgError
    except NotSquareImgError as e:
        print(e)

    return channels, img_size


def show_dataset_sample(data: datasets, classes: dict, show_fig: bool = True) -> None:
    """
    学習データの表示。9個のサンプルを表示する。
    """
    if not(show_fig):
        return
    plt.figure(figsize=(8, 8))
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        image, label = data[i]
        img = image.permute(1, 2, 0)  # 軸の入れ替え (C,H,W) -> (H,W,C)
        plt.imshow(img)
        ax.set_title(classes[label])
        # 枠線消し
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def confirm_scheduler(scheduler: CosineLRScheduler, show_fig: bool = True) -> None:
    """
    スケジューラによる学習率の変化の確認用
    """
    if not(show_fig):
        return
    lrs = []
    for i in range(scheduler.t_initial):
        if i == 0:
            print(f"warmup_lr_init: {scheduler._get_lr(i)}")
        lrs.append(scheduler._get_lr(i))
        if i == 30:
            print(f"finish warmup: {scheduler._get_lr(i)}")
        elif i == scheduler.t_initial - 1:
            print(scheduler._get_lr(i))
    plt.plot(lrs)
    plt.show()


if __name__ == "__main__":
    #-------------------------#
    #          諸準備          #
    #-------------------------#
    device = set_device()  # デバイスの指定
    fix_seed(device=device)  # 乱数シードの固定


    #------------------------------#
    #          モデルの定義          #
    #------------------------------#
    # /Users/[user_name]/.cache/torch/hub/checkpoints/ に保存される
    model = vit.vit_b_16().to(device)
    pretrained_model = vit.vit_b_16(weights=vit.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    summary(model=model, input_size=(256, 3, 224, 224))
    """
    # vit_b_16のrecipe
    [Link](https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16)

    torchrun --nproc_per_node=8 train.py\
        --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
        --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
        --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
    """
    epochs = 300
    batch_size = 512

    learning_rate = 0.003
    weight_decay = 0.3
    lr_warmup_epochs = 30
    lr_warmup_decay = 0.033

    label_smoothing_epsilon = 0.11

    dropout = 0.1

    mixup_alpha = 0.2
    auto_augment = "ra"  # RandAugment: Practical automated data augmentation with a reduced search space

    # 入力画像の前処理情報、preprocess(img)でいい
    preprocess = vit.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    optimizer = torch.optim.Adam(
        params=pretrained_model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,  # The initial number of epochs.
        warmup_t=lr_warmup_epochs,  # The number of warmup epochs.
        warmup_prefix=True, # If set to `True`, then every new epoch number equals `epoch = epoch - warmup_t`.
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_epsilon)  # Label Smoothingありの損失関数
    confirm_scheduler(scheduler, show_fig=False)


    #----------------------------------#
    #          データセットの用意         #
    #----------------------------------#
    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_data  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    # データセットの分割
    # train:val:test = 8:1:1
    num_val  = int(len(test_data) * 0.5)
    num_test = len(test_data) - num_val
    val_data, test_data = random_split(test_data, [num_val, num_test])
    
    # データローダーの作成
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    confirm_dataset(train_data, val_data, test_data)
    channels, img_size = get_img_info(train_dataloader)

    # CIFAR 10の全クラス
    classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    show_dataset_sample(train_data, classes, show_fig=False)
