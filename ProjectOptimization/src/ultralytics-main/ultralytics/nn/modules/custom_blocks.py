"""自定义模块：ADown / DyRFAUp / SCSA。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADown(nn.Module):
    """轻量非对称下采样模块。"""

    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.conv1 = nn.Conv2d(c1, c_, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.conv2 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        a = self.act(self.bn1(self.conv1(x)))
        b = self.pool(x)
        b = self.act(self.bn2(self.conv2(b)))
        return torch.cat([a, b], dim=1)


class DySample(nn.Module):
    """动态上采样模块（学习偏移后重采样）。"""

    def __init__(self, c, scale=2):
        super().__init__()
        self.scale = scale
        self.offset = nn.Conv2d(c, 2 * scale * scale, 1, 1, 0)

    def forward(self, x, return_offset=False):
        b, c, h, w = x.shape
        s = self.scale
        off = self.offset(x).view(b, s * s, 2, h, w)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1)[None, None]

        off_norm = torch.zeros_like(off)
        off_norm[:, :, 0] = off[:, :, 0] / max(w - 1, 1)
        off_norm[:, :, 1] = off[:, :, 1] / max(h - 1, 1)
        grid = base + off_norm.permute(0, 1, 3, 4, 2)

        xs = []
        for i in range(s * s):
            xi = F.grid_sample(x, grid[:, i], mode="bilinear", padding_mode="border", align_corners=True)
            xs.append(xi)

        y = torch.stack(xs, dim=2).view(b, c * s * s, h, w)
        y = F.pixel_shuffle(y, s)
        return (y, off) if return_offset else y


class RFAConv(nn.Module):
    """RFAConv简化版：多尺度DW卷积 + 注意力融合。"""

    def __init__(self, c, reduction=4):
        super().__init__()
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)

        hidden = max(c // reduction, 8)
        self.w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, 1, 0),
            nn.Softmax(dim=1),
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        a = self.dw3(x)
        b = self.dw5(x)
        w = self.w(x)
        y = a * w[:, 0:1] + b * w[:, 1:2]
        y = self.pw(y)
        return self.act(self.bn(y))


class DyRFAUp(nn.Module):
    """动态上采样与恒等上采样融合，并进行门控细化。"""

    def __init__(self, c, scale=2, mode="nearest", reduction=8):
        super().__init__()
        self.up_dy = DySample(c, scale=scale)
        self.up_id = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.refine = nn.Sequential(
            RFAConv(c),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        hidden = max(c // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y_id = self.up_id(x)
        y_dy, off = self.up_dy(x, return_offset=True)
        y = self.fuse(torch.cat([y_dy, y_id], dim=1))
        r = self.refine(y)

        off_mag = off.abs().mean(dim=(1, 2, 3, 4), keepdim=True).squeeze(-1)
        off_gate = torch.sigmoid(5.0 * off_mag)
        g = self.gate(y) * off_gate
        return y_id + r * g


class SCSA(nn.Module):
    """SCSA轻量注意力模块。"""

    def __init__(self, c, heads=4, reduction=4):
        super().__init__()
        self.heads = heads
        self.dk = max(c // heads, 16)

        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.sproj = nn.Conv2d(c, c, 1, 1, 0, bias=False)

        self.q = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(self.dk * heads, c, 1, 1, 0, bias=False)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // reduction, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.sproj(self.dw3(x) + self.dw5(x))
        x1 = x + s

        q = F.adaptive_avg_pool2d(self.q(x1), 1).flatten(2)
        k = F.adaptive_avg_pool2d(self.k(x1), 1).flatten(2)
        v = F.adaptive_avg_pool2d(self.v(x1), 1).flatten(2)

        b = x.size(0)
        q = q.view(b, self.heads, self.dk, 1)
        k = k.view(b, self.heads, self.dk, 1)
        v = v.view(b, self.heads, self.dk, 1)

        attn = torch.softmax((q * k).sum(dim=2, keepdim=True), dim=-1)
        y = self.out((attn * v).view(b, self.heads * self.dk, 1, 1))
        return x1 + y * self.gate(x1)


__all__ = ["ADown", "DySample", "RFAConv", "DyRFAUp", "SCSA"]
