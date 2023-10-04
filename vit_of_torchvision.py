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
            print(f"warmup_lr_init: {scheduler._get_lr(i)[0]}")
        lrs.append(scheduler._get_lr(i))
        if i == 30:
            print(f"finish warmup : {scheduler._get_lr(i)[0]}")
        elif i == scheduler.t_initial - 1:
            print(f"final lr      : {scheduler._get_lr(i)[0]}\n")
    plt.plot(lrs)
    plt.show()


def train(model, criterion, optimizer, dataloader: DataLoader) -> Tuple[float, float]:
    """
    学習用関数。1エポック間の学習について記述する。
    """


def validation(model, criterion, dataloader: DataLoader) -> Tuple[float, float]:
    """
    検証用関数。`train()`後に配置する
    """


def test(model, criterion, dataloader: DataLoader) -> Tuple[float, float]:
    """
    テスト用関数。全エポック終了後に配置する。`validation()`を流用
    """


def save_lc_of_loss(train_loss_list: list, val_loss_list: list, date_now: str) -> None:
    """
    損失の学習曲線を保存する。学習時と検証時とを同時に描画する。
    """


def save_lc_of_acc(train_loss_list: list, val_loss_list: list, date_now: str) -> None:
    """
    精度の学習曲線を保存する。学習時と検証時とを同時に描画する。
    """


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
    """
    # vit_b_16のrecipe
    [Link](https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16)

    torchrun --nproc_per_node=8 train.py\
        --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
        --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
        --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
    """
    pretrained_model = vit.vit_b_16(weights=vit.ViT_B_16_Weights.IMAGENET1K_V1).to(device)

    # 最終層の取り換え
    pretrained_model.heads[0] = nn.Linear(in_features=768, out_features=10, bias=True)
    nn.init.constant_(pretrained_model.heads[0].weight, 0)  # Zero-initialize
    nn.init.constant_(pretrained_model.heads[0].bias, 0) # Zero-initialize

    epochs = 300
    batch_size = 128

    learning_rate = 0.001
    weight_decay = 0

    lr_warmup_epochs = 5
    lr_warmup_init   = 0.

    label_smoothing_epsilon = 0.11

    # モデル構造の確認
    summary(model=pretrained_model, input_size=(batch_size, 3, 224, 224))
    print()

    optimizer = torch.optim.SGD(
        params=pretrained_model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        momentum=0.9
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,  # The initial number of epochs.
        warmup_t=lr_warmup_epochs,  # The number of warmup epochs.
        warmup_lr_init=lr_warmup_init, 
        warmup_prefix=True, # If set to `True`, then every new epoch number equals `epoch = epoch - warmup_t`.
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_epsilon)  # Label Smoothingありの損失関数
    confirm_scheduler(scheduler, show_fig=False)


    #----------------------------------#
    #          データセットの用意         #
    #----------------------------------#
    # 入力画像の前処理情報、preprocess(img)でいい
    preprocess = vit.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    test_data  = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
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


    #######################################
    #          ファインチューニング          #
    #######################################
    print("\033[44mTraining Step\033[0m")

    for t in range(epochs):
        print(f"Epoch {t+1}\n----------------------------------------------------------------")

        print("\033[34mTrain\033[0m")

        print("\033[34mValidation\033[0m")

    print("\033[44mTest Step\033[0m")

    
    ################################
    #          成果物の保存          #
    ################################