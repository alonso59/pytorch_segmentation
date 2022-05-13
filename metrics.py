import torch
import torch.nn.functional as F
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self, class_index):
        super().__init__()
        self.class_index = class_index
    
    @property
    def __name__(self):
        return "Accuracy"

    def forward(self, y_pred, y):
        num_classes = y_pred.shape[1]
        true_1_hot = torch.eye(num_classes)[y.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(y_pred.type())
        if num_classes > 1:
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).unsqueeze(1)
        else:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred).unsqueeze(1)
        tp = torch.sum(true_1_hot == y_pred)
        score = tp / true_1_hot.reshape(-1).shape[0]
        return score


class mIoU(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    
    property
    def __name__(self):
        return "IoU"

    def forward(self, y_pred, y, eps=1e-5):
        num_classes = y_pred.shape[1]
        true_1_hot = torch.eye(num_classes)[y.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(y.type()).to(self.device)
        probas = F.softmax(y_pred, dim=1).to(self.device)
        dims = (0, 2, 3)
        mult = (probas * true_1_hot).to(self.device)
        sum = (probas + true_1_hot).to(self.device)
        intersection = torch.sum(mult, dim=dims)
        cardinality = torch.sum(sum, dim=dims)
        union = cardinality - intersection
        iou_score = (intersection / (union + eps))
        return iou_score.cpu().detach().numpy()


class IoU_binary(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    @property
    def __name__(self):
        return "IoU"

    def forward(self, y_pred, y, eps=1e-5):
        true_1_hot = torch.eye(2, device=self.device)[y.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(y_pred)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        dims = (0,) + tuple(range(2, y.ndimension()))
        mult = (probas * true_1_hot)
        sum = (probas + true_1_hot)
        intersection = torch.sum(mult, dim=dims)
        cardinality = torch.sum(sum, dim=dims)
        union = cardinality - intersection
        iou_score = (intersection / (union + eps))
        return iou_score.cpu().detach().numpy()