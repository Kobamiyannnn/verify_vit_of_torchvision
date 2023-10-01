import torch
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

def scheduler():
    """
    `optimizer`の`lr`までwarm upで上がった後、`scheduler`の`lr_min`までCosineで向かっていく
    """
    model = torch.nn.Linear(1, 1) ## 適当なモデル
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineLRScheduler(optimizer, t_initial=200, lr_min=1e-4, 
                                  warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)

    lrs = []
    for i in range(200):
        # lrs.append(scheduler.get_epoch_values(i))
        lrs.append(scheduler._get_lr(i))
        if i == 199:
            print(scheduler._get_lr(i))
    plt.plot(lrs)
    plt.show()

if __name__ == "__main__":
    scheduler()
