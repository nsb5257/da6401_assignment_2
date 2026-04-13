"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth",
                 unet_path: str = "unet.pth", use_bn: bool = True):
        super().__init__()
        import gdown
        gdown.download(id="1px9d3hh2hUYFMclVIlIZ0bViHGBHzf6d", output=classifier_path, quiet=False)
        gdown.download(id="1FXS53aHA6L6xWbyGU0ObX6JkOTEIdGT", output=localizer_path, quiet=False)
        gdown.download(id="19a3xxVIWxthrB1jvih0eB1erFBTDBtvB", output=unet_path, quiet=False)

        self.classifier_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels, use_bn=use_bn)
        self.localizer_model = VGG11Localizer(in_channels=in_channels, use_bn=use_bn)
        self.unet_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, use_bn=use_bn)

        def load_ckpt(model, path):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state)

        load_ckpt(self.classifier_model, classifier_path)
        load_ckpt(self.localizer_model, localizer_path)
        load_ckpt(self.unet_model, unet_path)
        self.shared_encoder = self.classifier_model.encoder

    def forward(self, x: torch.Tensor):
        bottleneck, features = self.shared_encoder(x, return_features=True)

        c = self.classifier_model.avgpool(bottleneck)
        c = torch.flatten(c, 1)
        class_out = self.classifier_model.classifier(c)

        l = self.localizer_model.avgpool(bottleneck)
        l = torch.flatten(l, 1)
        loc_out = self.localizer_model.regressor(l)

        d = self.unet_model.up1_trans(bottleneck)
        d = self.unet_model._pad_and_cat(d, features['block5'])
        d = self.unet_model.up1_conv(d)
        d = self.unet_model.up2_trans(d)
        d = self.unet_model._pad_and_cat(d, features['block4'])
        d = self.unet_model.up2_conv(d)
        d = self.unet_model.up3_trans(d)
        d = self.unet_model._pad_and_cat(d, features['block3'])
        d = self.unet_model.up3_conv(d)
        d = self.unet_model.up4_trans(d)
        d = self.unet_model._pad_and_cat(d, features['block2'])
        d = self.unet_model.up4_conv(d)
        d = self.unet_model.up5_trans(d)
        d = self.unet_model._pad_and_cat(d, features['block1'])
        d = self.unet_model.up5_conv(d)
        seg_out = self.unet_model.final_conv(d)

        return {'classification': class_out, 'localization': loc_out, 'segmentation': seg_out}