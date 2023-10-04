import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vision_transformer as vit
from torchvision.models.vision_transformer import ViT_B_16_Weights
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt
from torchsummary import summary

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n")

model = vit.vit_b_16().to(device)  # /Users/[user_name]/.cache/torch/hub/checkpoints/ に保存される

pretrained_model = vit.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
"""
# vit_b_16のrecipe
[Link](https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16)

torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema

"""

# 入力画像の前処理情報、preprocess(img)でいい
preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
print(preprocess)

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


optimizer = torch.optim.Adam(
    params=model.parameters(), 
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


def confirm_scheduler(epochs, scheduler: CosineLRScheduler):
    lrs = []
    for i in range(epochs):
        if i == 0:
            print(f"warmup_lr_init: {scheduler._get_lr(i)}")
        lrs.append(scheduler._get_lr(i))
        if i == 30:
            print(f"finish warmup: {scheduler._get_lr(i)}")
        elif i == epochs - 1:
            print(scheduler._get_lr(i))
    plt.plot(lrs)
    plt.show()

confirm_scheduler(epochs, scheduler)