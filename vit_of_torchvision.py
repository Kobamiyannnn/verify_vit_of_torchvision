import torch
import torchvision.models.vision_transformer as vit
from torchvision.models.vision_transformer import ViT_B_16_Weights

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n")

model = vit.vit_b_16()

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

epochs = 300
batch_size = 512

adam_lr = 0.003
adam_wd = 0.3
adam_lr_scheduler = "cosineannealing"
