import os
import sys
import yaml
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from training.metrics import MIoU
from training.dataset import ImagesFromFolder

def visualize(n, image, mask, pr_mask, path_save, name, metric_dict):
    """PLot images in one row."""
    figure, ax = plt.subplots(nrows=n, ncols=3)
    figure.set_figwidth(12)
    figure.set_figheight(10)
    for i in range(n):
        ax[i, 0].imshow(image[i, :, :], cmap='gray')
        ax[0, 0].title.set_text('Test image')
        ax[i, 0].axis('off')
        ax[i, 1].imshow(mask[i, :, :], cmap='jet')
        ax[0, 1].title.set_text('Test mask')
        ax[i, 1].axis('off')
        ax[i, 2].imshow(pr_mask[i, :, :], cmap='jet')
        ax[0, 2].title.set_text(f'Prediction \n{metric_dict[0]}')
        ax[i, 2].title.set_text(f'{metric_dict[i]}')
        ax[i, 2].axis('off')
    figure.suptitle(name)
    plt.savefig(path_save + "/" + name + '_' + str(np.random.randint(0, 100)) + ".png")
    plt.show()


def main(cfg):
    paths = cfg['paths']
    train_imgdir = paths['train_imgdir']
    train_mskdir = paths['train_mskdir']
    val_imgdir = paths['val_imgdir']
    val_mskdir = paths['val_mskdir']
    """ CUDA device """
    device = torch.device("cuda")
    path = 'logs/2022-07-01_00_04_00/checkpoints/model.pth'
    best_model = torch.load(path)
    imgs = []
    mask_true = []
    prds_msk = []
    j = 3
    iou = MIoU(device)
    # np.random.seed(20)  # 2, 10, 20, 42, 32
    res_metric = {}
    val_ds = ImagesFromFolder(image_dir=val_imgdir,
                              mask_dir=val_mskdir,
                              transform=None,
                              )

    for i in range(j):
        randint = np.random.randint(low=0, high=len(val_ds))
        image, mask = val_ds[randint]
        image1 = torch.tensor(image, dtype=torch.float, device=device)
        image1 = image1.unsqueeze(0)
        mask1 = torch.tensor(mask, dtype=torch.long, device=device)
        mask1 = mask1.unsqueeze(0)
        pr_mask = best_model(image1)
        metric = iou(pr_mask, mask1)
        if metric.shape[0] == 3:
            print_metric = f"BG:{metric[0]:2.3f}, OPL:{metric[1]:2.3f}, EZ:{metric[2]:2.3f}"
        elif metric.shape[0] == 4:
            print_metric = f"BG:{metric[0]:2.3f}, OPL:{metric[1]:2.3f}, ELM:{metric[2]:2.3f}, EZ:{metric[3]:2.3f}"
        else:
            print("Invalid numb of class")
            sys.exit()
        res_metric[i] = print_metric
        print(print_metric)
        print(np.mean(metric))
        pr_mask = F.softmax(pr_mask, dim=1)
        pr_mask = torch.argmax(pr_mask, dim=1)
        pr_mask = (pr_mask.squeeze().cpu().float().detach().numpy())
        image1 = image1.squeeze(0).squeeze(0).cpu().detach().numpy()
        mask1 = mask1.squeeze(0).squeeze(0).cpu().detach().numpy()
        imgs.append(image1)
        mask_true.append(mask1)
        prds_msk.append(pr_mask)

    path_save = os.path.split(path)[0]
    visualize(n=len(imgs), image=np.array(imgs),
              mask=np.array(mask_true), pr_mask=np.array(prds_msk),
              path_save=path_save, name='unet', metric_dict=res_metric
              )


if __name__ == '__main__':
    with open('configs/oct.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(cfg)