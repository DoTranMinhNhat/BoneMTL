# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsample → concat skip → 2x ConvBNReLU.

    Args:
        in_ch   : channels từ decoder bước trước
        skip_ch : channels từ encoder skip connection
        out_ch  : channels output
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True,
        )
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=True,
            )
        return self.conv(torch.cat([x, skip], dim=1))


class BoneMTL(nn.Module):
    """
    Multi-Task Learning model cho phân tích bone tumor từ ảnh X-quang.

    Kiến trúc:
        Shared encoder : ResNet50 pretrained (ImageNet)
        Classification : hierarchical head 3 tầng
        Segmentation   : U-Net decoder + Deep Supervision (aux3, aux2)

    Deep Supervision: auxiliary loss tại dec3 và dec2 giúp gradient
    flow tốt hơn qua decoder trong quá trình train.

    Args:
        num_tumor_types : số loại u ở tier 3 (mặc định là 9)
        pretrained      : dùng ImageNet pretrained weights
    """

    def __init__(self, num_tumor_types: int = 9, pretrained: bool = True):
        super().__init__()

        # Encoder — ResNet50
        backbone  = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Classification head
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.fc_tier1 = nn.Linear(512, 1)
        self.fc_tier2 = nn.Linear(512, 1)
        self.fc_tier3 = nn.Linear(512, num_tumor_types)

        # Segmentation decoder — U-Net
        self.dec4 = DecoderBlock(2048, 1024, 256)
        self.dec3 = DecoderBlock(256,  512,  128)
        self.dec2 = DecoderBlock(128,  256,  64)
        self.dec1 = DecoderBlock(64,   64,   32)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(32, 16),
        )
        self.seg_out = nn.Conv2d(16, 1, kernel_size=1)

        # Deep Supervision auxiliary heads
        self.aux3 = nn.Conv2d(128, 1, kernel_size=1)  # từ dec3
        self.aux2 = nn.Conv2d(64,  1, kernel_size=1)  # từ dec2

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, 3, H, W)

        Returns:
            dict:
                tier1 : (B, 1)        logits
                tier2 : (B, 1)        logits
                tier3 : (B, 9)        logits
                mask  : (B, 1, H, W)  logits — main output
                aux3  : (B, 1, H, W)  logits — auxiliary (training only)
                aux2  : (B, 1, H, W)  logits — auxiliary (training only)
        """
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Classification
        feat  = self.cls_head(self.gap(e4).flatten(1))
        tier1 = self.fc_tier1(feat)
        tier2 = self.fc_tier2(feat)
        tier3 = self.fc_tier3(feat)

        # Segmentation decoder
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        seg = self.seg_out(self.dec0(d1))

        # Auxiliary outputs — upsample về input size
        aux3 = F.interpolate(
            self.aux3(d3), size=x.shape[2:], mode='bilinear', align_corners=True
        )
        aux2 = F.interpolate(
            self.aux2(d2), size=x.shape[2:], mode='bilinear', align_corners=True
        )

        return {
            'tier1': tier1, 'tier2': tier2, 'tier3': tier3,
            'mask':  seg,
            'aux3':  aux3,
            'aux2':  aux2,
        }