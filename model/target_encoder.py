from torch import nn
from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from torch import Tensor
from torch.nn import functional as F


class EncoderT(nn.Module):
    def __init__(self, variant, pretrained_backbone):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)

    def forward(self, src, downsample_ratio: float = 1):
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        return f1, f2, f3, f4

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
