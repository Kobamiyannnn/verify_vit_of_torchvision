import pandas as pd
from vit_of_torchvision import save_lc_of_loss, save_lc_of_acc
import matplotlib.pyplot as plt

acc_df = pd.read_csv("./results/2023-10-10T20:59:56/acc_in_learn_2023-10-10T20:59:56.csv", header=0, names=["train_acc", "val_acc"])
loss_df = pd.read_csv("./results/2023-10-10T20:59:56/loss_in_learn_2023-10-10T20:59:56.csv", header=0, names=["train_loss", "val_loss"])

save_lc_of_acc(list(acc_df["train_acc"]), list(acc_df["val_acc"]), dir_path="./results/2023-10-10T20:59:56")
save_lc_of_loss(list(loss_df["train_loss"]), list(loss_df["val_loss"]), dir_path="./results/2023-10-10T20:59:56")