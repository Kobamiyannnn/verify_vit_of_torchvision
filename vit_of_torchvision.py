import torch
import torchvision.models.vision_transformer as vit

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n")

model = vit.vit_b_16()
print(model)