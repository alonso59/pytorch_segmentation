import os
import sys
import cv2
import yaml
import torch
import pandas as pd
import numpy as np
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import matplotlib

from matplotlib import cm
from PIL import Image
from utils.utils import get_filenames, create_dir
from patchify import patchify, unpatchify

from training.loss import *
from training.trainer import eval
from training.metrics import MIoU
from training.dataset import ImagesFromFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
from monai.metrics.confusion_matrix import get_confusion_matrix
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses.dice import DiceFocalLoss

def gray_gamma(img, gamma):
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')

def tv_denoising(img, alpha):
    gray = img / 255.
    out = denoise_tv_chambolle(gray, weight=alpha)
    out = out * 255
    return out.astype('uint8')

def get_segmentation(self, model, img, mode, gamma=1, alpha=0.0001):
    img_in = gray_gamma(img, gamma=gamma)
    img_in = tv_denoising(img_in, alpha=alpha)
    
    if mode == 'large':
        shape_image_x = img_in.shape
        image_x = F.interpolate(torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0).float(), (self.imgh, self.imgw), mode='bilinear', align_corners=False).squeeze().numpy()
        pred = predict(model, image_x, self.device)
        preds = F.interpolate(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(), (shape_image_x[0], shape_image_x[1]), mode='nearest').squeeze().numpy()
        img_out = img_in
    
    if mode == 'patches':
        pady = 0
        padx = 0
        if img.shape[0] == 496:
            pady = 8
        if img.shape[1] == 1000:
            padx = 12
        large_image = np.pad(img_in, [(pady, ), (padx, )], 'constant', constant_values=0)
        patches_images = patchify(large_image, (self.imgh, self.imgw), step=self.imgw)
        predictions = []
        for i in range(patches_images.shape[0]):
            for j in range(patches_images.shape[1]):
                image_x = patches_images[i, j, :, :]
                pred = predict(model, image_x, self.device)
                pred = Image.fromarray(pred.astype('uint8'))
                predictions.append(np.array(pred))
        predictions = np.array(predictions)
        predictions = np.reshape(predictions, patches_images.shape)
        rec_img = unpatchify(patches=patches_images, imsize=(self.imgh * predictions.shape[0], self.imgw * predictions.shape[1]))
        preds = unpatchify(patches=predictions, imsize=(self.imgh * predictions.shape[0], self.imgw * predictions.shape[1]))
        preds = preds[pady:img.shape[0]+pady, padx:img.shape[1]+padx]
        img_out = rec_img[pady:img.shape[0]+pady, padx:img.shape[1]+padx]

    if mode == 'slices':
        predictions = []
        if not self.per_batch:
            for i in range(self.imgw, img_in.shape[1]+self.imgw, self.imgw):
                image_x = img_in[:, i - self.imgw:i]
                pred = predict(model, image_x, self.device)
                predictions.append(pred.astype('uint8'))
        else:
            img_pred = []
            for i in range(self.imgw, img_in.shape[1]+self.imgw, self.imgw):
                image_x = img_in[:, i - self.imgw:i]
                img_pred.append(image_x)
            img_pred = np.array(img_pred)
            predictions = predict(self.model, img_pred, self.device)
        predictions = np.array(predictions)
        preds = np.hstack(predictions)
        img_out = img_in
    
    # prepare output
    shape_1 = (preds.shape[0], preds.shape[1], 3)
    pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=preds.max())
    for idx in range(1, int(preds.max())+1):
        pred_rgb[..., 0] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[0], pred_rgb[..., 0])
        pred_rgb[..., 1] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[1], pred_rgb[..., 1])
        pred_rgb[..., 2] = np.where(preds == idx, cm.hsv(norm(idx), bytes=True)[2], pred_rgb[..., 2])

    # output
    img_overlay = Image.fromarray(img_out)
    pred_overlay = Image.fromarray(pred_rgb)
    img_overlay = img_overlay.convert("RGBA")
    pred_overlay = pred_overlay.convert("RGBA")
    overlay = Image.blend(img_overlay, pred_overlay, 0.4)
    overlay = np.array(overlay)
    return preds, pred_rgb, overlay

def predict_1(model, x_image, y_mask, device):
    MEAN =  0.1338 # 0 # 0.1338  # 0.1338 0.13505013393330723
    STD =  0.1466 # 1 # 0.1466  # 0.1466 0.21162075769722669

    iou_fn = MIoU(activation='softmax', ignore_background=True, device=device)

    transforms = T.Compose([
                T.Normalize(mean=MEAN, std=STD) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
    ])
    n_dimention = np.ndim(x_image)
    
    image = np.expand_dims(x_image, axis=-1)
    
    # image = np.repeat(image, 3, axis=-1)

    image = transforms(image=image)
    if n_dimention == 2:
        image = image['image'].transpose((2, 0, 1))
    elif n_dimention == 3:
        image = image['image'].transpose((0, 3, 1, 2))
    image = torch.tensor(image, dtype=torch.float, device=device)

    if torch.Tensor.dim(image) == 3:
        image = image.unsqueeze(0)

    y_pred = model(image)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)

    iou = iou_fn(y_pred, y_mask)
    iou = np.where(iou <= 1e-1, 1, iou)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.squeeze(0).detach().cpu().numpy()

    if n_dimention == 2:
        y_pred = y_pred.squeeze(0)
    elif n_dimention == 3:
        y_pred = y_pred.squeeze(1)

    y_pred = y_pred.detach().cpu().numpy()

    return y_pred


def predict(model, x_image, y_mask):
    device = torch.device("cuda")

    iou_fn = MIoU(activation='softmax', ignore_background=True, device=device)

    transforms = T.Compose([
                T.Normalize(mean=0.1338, std=0.1466) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
    ])
    image = np.expand_dims(x_image, axis=-1)
    y_mask = torch.tensor(y_mask, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    # image = np.repeat(image, 3, axis=-1)

    image = transforms(image=image)
    image = image['image'].transpose((2, 0, 1))
    image = torch.tensor(image, dtype=torch.float, device=device)
    image = image.unsqueeze(0)

    y_pred = model(image)
    iou = iou_fn(y_pred, y_mask)
    iou = np.where(iou <= 1e-1, 1, iou)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.squeeze(0).detach().cpu().numpy()
    # print(iou.mean())
    return y_pred, iou

def run_evaluation(model, images, masks, image_sizeh, image_sizew):
    device = torch.device("cuda")
    loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.6, class_weights=[1 for _ in range(5)])
    # loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True)
    # loss_fn = CrossEntropyLoss(device=device)

    val_transforms = T.Compose([
        T.Resize(image_sizeh, image_sizew),
        T.Normalize(mean=0.1338, std=0.1466)
        ])

    val_ds = ImagesFromFolder(image_dir=images,
                              mask_dir=masks,
                              transform=val_transforms,
                              preprocess_input=None
                              )

    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=False
                            )

    loss_eval, iou_eval, pixel_acc_list, dice_list, precision_list, recall_list = eval(model, val_loader, loss_fn, device)

    div_iou = np.array(iou_eval).mean(axis=0)
    div_pxl = np.array(pixel_acc_list).mean(axis=0)
    div_dice = np.array(dice_list).mean(axis=0)
    div_pre = np.array(precision_list).mean(axis=0)
    div_rec = np.array(recall_list).mean(axis=0)

    print(f'IoU: \n', div_iou, div_iou)
    print(f'P. Acc: \n', div_pxl)
    print(f'Dice: \n', div_dice, div_dice)
    print(f'Prec.: \n', div_pre, div_pre)
    print(f'Recall: \n', div_rec, div_rec)

    print('Loss Eval: ', loss_eval)

def main():
    base_path = 'logs/2022-09-18_22_35_46'
    model_path = os.path.join(base_path, 'checkpoints/model.pth')

    with open(os.path.join(base_path, 'experiment_cfg.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    paths = cfg['paths']
    base = paths['base']
    general = cfg['general']
    image_sizeh = general['img_sizeh']
    image_sizew = general['img_sizew']
    val_imgdir = os.path.join(base, paths['val_imgdir'])
    val_mskdir = os.path.join(base, paths['val_mskdir'])

    test_imgdir = os.path.join(base, paths['test_imgdir'])
    test_mskdir = os.path.join(base, paths['test_mskdir'])

    pred_imgdir = os.path.join(base, paths['save_testimg'])
    pred_mskdir = os.path.join(base, paths['save_testmsk'])
    pred_predsdir = os.path.join(base, paths['save_testpred'])

    create_dir(pred_imgdir)
    create_dir(pred_mskdir)
    create_dir(pred_predsdir)
    create_dir(os.path.join(pred_predsdir, 'overlay'))

    files = get_filenames(test_imgdir, 'png')
    filesM = get_filenames(test_mskdir, 'png')
    
    model = torch.load(model_path)

    # run_evaluation(model, val_imgdir, val_mskdir, image_sizeh, image_sizew)

    iou = []
    fileImage = []

    for im, mk in zip(files, filesM):
        # iou_item = get_segmentation_patches(model, im, mk, pred_imgdir, pred_mskdir, pred_predsdir)
        iou_item = get_segmentation_large(model, im, mk, pred_imgdir, pred_mskdir, pred_predsdir, imgw=496, imgh=496, gamma=1.5, alpha=0.01)
        iou.append(iou_item)
        fileImage.append(os.path.split(im)[1])

    iou = np.array(iou)

    # df2 = pd.DataFrame({'file': fileImage,
    #                     'iou': iou,
    #                     })
    # df2.to_csv('predictions.csv')
    # iou.sort()
    hmean = np.mean(iou, axis=0)
    hstd = np.std(iou, axis=0)
    # _, q1, q3 = np.percentile(iou, 50), np.percentile(iou, 25), np.percentile(iou, 75)
    # sigma = hstd
    # mu = hmean
    # iqr = 1.5 * (q3 - q1)
    # x1 = np.linspace(q1 - iqr, q1)
    # x2 = np.linspace(q3, q3 + iqr)
    # pdf1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
    # pdf2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x2 - mu)**2 / (2 * sigma**2))

    print(f'Mean per class:{hmean}, Std per class:{hstd}')
    print(f'Mean :{np.mean(iou)}, Std:{np.std(iou)}')
    # pdf = stats.norm.pdf(iou, hmean, hstd)
    # pl.plot(iou, pdf, '-o', label=f'Mean:{hmean:0.3f}, Std:{hstd:0.3f}, Q1:{q1:0.3f}, Q3:{q3:0.3f}')

    arran = np.linspace(0.5, 1, num=(len(iou)//10))
    plt.hist(iou.mean(), bins=arran, edgecolor='black')
    # pl.fill_between(x1, pdf1, 0, alpha=.6, color='green')
    # pl.fill_between(x2, pdf2, 0, alpha=.6, color='green')
    # plt.xlim([0.4, 1.1])
    plt.xlabel('IoU', fontsize=18, fontweight='bold')
    plt.ylabel('No. Images', fontsize=18, fontweight='bold')
    # plt.legend(loc='best')
    plt.savefig('train.png')


if __name__ == '__main__':
    main()