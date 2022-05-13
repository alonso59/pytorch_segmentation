from networks.swin_unet import SwinUnet
from networks.unet import UNet
import torch.nn as nn
from summary import summary
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class SegmentationModels(nn.Module):
    def __init__(self, device, in_channels, img_size, n_classes=1) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.img_size = img_size
        self.n_classes = n_classes

    def summary(self, logger=None):
        summary(self.model, input_size=(self.in_channels, self.img_size, self.img_size), batch_size=-1, logger=logger)

    def UNet(self, feature_start=64, layers=4, bilinear=False, dropout=0.0, kernel_size=3, stride=1, padding=1):

        self.model = UNet(
            num_classes=self.n_classes,
            input_channels=self.in_channels,
            num_layers=layers,
            features_start=feature_start,
            bilinear=bilinear,
            dp=dropout,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            stride=stride
        ).to(self.device)

        return self.model, None

    def SwinUnet(self, 
                pretrain=True,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.1,
                ):

        self.model = SwinUnet(
                in_chans=self.in_channels, 
                img_size=self.img_size, 
                num_classes=self.n_classes, 
                zero_head=False,
                patch_size=4,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                drop_path_rate=drop_path_rate,
        ).to(self.device)
        
        if pretrain:
            self.model.state_dict()
            self.model.load_from("../pretrained/swin_tiny_patch4_window7_224.pth", self.device)
        return self.model, None


    def UNet_imagenet(self, encoder_name="resnet18", encoder_weights="imagenet"):
        # self.model = smp.Unet(in_channels=self.in_channels, classes=self.n_classes).to(self.device)
        self.model = smp.Unet(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=self.in_channels ,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=self.n_classes,                      # model output channels (number of classes in your dataset)
        ).to(self.device)
        
        preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')

        return self.model, preprocess_input
    """
    you can add your own network here
    .
    .
    .
    """
