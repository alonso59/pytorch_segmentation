from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks as drawer 
import sys
import numpy as np
from torchvision import transforms
from torch.nn import functional as F

class TensorboardWriter(object):

    def __init__(self, metric, name_dir):

        super(TensorboardWriter).__init__()
        self.writer = SummaryWriter(log_dir=name_dir)
        self.metric = metric

    def per_epoch(self, train_loss, val_loss, train_metric, val_metric, step):
        results_loss = {'Train': train_loss, 'Val': val_loss}
        results_metric = {'Train': train_metric, 'Val': val_metric}
        self.writer.add_scalars("Loss", results_loss, step)
        self.writer.add_scalars(
            f'IoU', results_metric, step)

    def per_iter(self, loss, metric, step, name):
        self.writer.add_scalar(f"{name}/Loss", loss, step)
        self.writer.add_scalar(f'{name}/IoU', metric, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)

    def save_graph(self, model, loader):
        self.writer.add_graph(model, loader)

    def save_text(self, tag, text_string):
        self.writer.add_text(tag=tag, text_string=text_string)

    def save_images(self, x, y, y_pred, step, device):
        pick1 = np.random.randint(1, len(x)-5)
        pick2 = pick1 - 4 if pick1 >= 4 else 1
        gt = image_tensorboard(y[:3, :, :], device)
        if y_pred.shape[1] == 1:
            pred = torch.sigmoid(y_pred[:3, :, :, :])
            pred = torch.round(pred)
        else:
            pred = torch.softmax(y_pred[:3, :, :, :], dim=1)
            pred = torch.argmax(pred, dim=1).unsqueeze(1)
        pred = image_tensorboard(pred, device)
        self.writer.add_images(f'Data',x[:3, :, :, :], step, dataformats='NCHW')
        self.writer.add_images(f'Ground truth', gt, step, dataformats='NCHW')
        self.writer.add_images(f'Prediction', pred.squeeze(1), step, dataformats='NCHW')

def image_tensorboard(img, device):
    img_rgb = torch.zeros((img.size(0), 3, img.size(2), img.size(3))).float().to(device)
    img_rgb[:, 0, :, :] = torch.where(img.squeeze(1) == 1, 1, 0)
    img_rgb[:, 1, :, :] = torch.where(img.squeeze(1) == 2, 1, 0)
    img_rgb[:, 2, :, :] = torch.where(img.squeeze(1) == 3, 1, 0)
    return img_rgb

def denormalize_imagenet(tensor):
    invTrans = transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225])
    return torch.clamp(invTrans(tensor), 0, 1)