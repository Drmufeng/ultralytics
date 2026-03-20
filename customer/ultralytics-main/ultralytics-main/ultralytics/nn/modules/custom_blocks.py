
import torch
import torch.nn as nn
import torch.nn.functional as F



class ADown(nn.Module):
    """Lightweight asymmetric downsampling block.
    Replace stride-2 Conv with a two-branch downsample + fuse.
    """
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
        a = self.act(self.bn1(self.conv1(x)))   # branch A: stride-2 conv
        b = self.pool(x)                        # branch B: maxpool + 1x1
        b = self.act(self.bn2(self.conv2(b)))
        return torch.cat([a, b], dim=1)



class DySample(nn.Module):
    """Ultra-light dynamic upsampler using learned offsets + grid_sample.
    drop-in replacement for nn.Upsample(scale_factor=2).
    """
    def __init__(self, c, scale=2):
        super().__init__()
        self.scale = scale
        self.offset = nn.Conv2d(c, 2 * scale * scale, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        off = self.offset(x)  # (B, 2*s*s, H, W)
        off = off.view(b, s * s, 2, h, w)  # (B, s*s, 2, H, W)

        # base grid in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=-1)          # (H, W, 2)
        base = base.unsqueeze(0).unsqueeze(1)         # (1, 1, H, W, 2)

        # normalize offsets to grid units
        off_norm = torch.zeros_like(off)
        off_norm[:, :, 0] = off[:, :, 0] / max(w - 1, 1)
        off_norm[:, :, 1] = off[:, :, 1] / max(h - 1, 1)
        grid = base + off_norm.permute(0, 1, 3, 4, 2)  # (B, s*s, H, W, 2)

        # sample each subpixel then pixel-shuffle merge
        xs = []
        for i in range(s * s):
            xi = F.grid_sample(
                x, grid[:, i],
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )
            xs.append(xi)

        y = torch.stack(xs, dim=2)      # (B, C, s*s, H, W)
        y = y.view(b, c * s * s, h, w)
        y = F.pixel_shuffle(y, s)       # (B, C, H*s, W*s)
        return y



class RFAConv(nn.Module):
    """RFAConv-lite: receptive-field attention style refinement.
    Multi-kernel depthwise conv + attention weights fuse -> pointwise conv.
    """
    def __init__(self, c, reduction=4):
        super().__init__()
        # multi-scale depthwise conv
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)

        hidden = max(c // reduction, 8)
        self.w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, 1, 0),   # weights for (dw3, dw5)
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        a = self.dw3(x)
        b = self.dw5(x)
        w = self.w(x)                    # (B, 2, 1, 1)
        y = a * w[:, 0:1] + b * w[:, 1:2]
        y = self.pw(y)
        return self.act(self.bn(y))



# ✅ 新的模块2：DyRFAUp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic upsample with learned offsets (can optionally return offsets)."""
    def __init__(self, c, scale=2):
        super().__init__()
        self.scale = scale
        self.offset = nn.Conv2d(c, 2 * scale * scale, 1, 1, 0)

    def forward(self, x, return_offset: bool = False):
        b, c, h, w = x.shape
        s = self.scale
        off = self.offset(x).view(b, s * s, 2, h, w)  # (B, s*s, 2, H, W)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1)[None, None]  # (1,1,H,W,2)

        off_norm = torch.zeros_like(off)
        off_norm[:, :, 0] = off[:, :, 0] / max(w - 1, 1)
        off_norm[:, :, 1] = off[:, :, 1] / max(h - 1, 1)
        grid = base + off_norm.permute(0, 1, 3, 4, 2)  # (B,s*s,H,W,2)

        xs = []
        for i in range(s * s):
            xi = F.grid_sample(
                x, grid[:, i],
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )
            xs.append(xi)

        y = torch.stack(xs, dim=2).view(b, c * s * s, h, w)
        y = F.pixel_shuffle(y, s)  # (B,C,H*s,W*s)

        if return_offset:
            return y, off
        return y


class RFAConv(nn.Module):
    """RFAConv-lite: multi-kernel DW conv + attention fuse -> PW conv."""
    def __init__(self, c, reduction=4):
        super().__init__()
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.pw  = nn.Conv2d(c, c, 1, 1, 0, bias=False)

        hidden = max(c // reduction, 8)
        self.w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        a = self.dw3(x)
        b = self.dw5(x)
        w = self.w(x)  # (B,2,1,1)
        y = a * w[:, 0:1] + b * w[:, 1:2]
        y = self.pw(y)
        return self.act(self.bn(y))


class DyRFAUp(nn.Module):
    """
    DyRFAUp++ (Fusion Module):
      Branch1: DySample (dynamic upsample)
      Branch2: Bilinear upsample (stable identity path)
      Fuse: concat + 1x1
      Refine: RFAConv
      Gate: SE gate * offset-aware scalar
      Out: interp + gate * refine
    """
    def __init__(self, c, scale=2, reduction=8):
        super().__init__()
        self.scale = scale
        self.up_dy = DySample(c, scale=scale)
        self.up_id = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        # fuse dy + id
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # refine with fusion-conv module (paper module B)
        self.refine = nn.Sequential(
            RFAConv(c),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # channel gate (SE style)
        hidden = max(c // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y_id = self.up_id(x)                      # identity/stable upsample
        y_dy, off = self.up_dy(x, return_offset=True)  # dynamic upsample + offsets

        y = self.fuse(torch.cat([y_dy, y_id], dim=1))
        r = self.refine(y)

        # offset-aware scalar (B,1,1,1): larger offsets -> allow more refine
        off_mag = off.abs().mean(dim=(1, 2, 3, 4), keepdim=True)  # (B,1,1,1,1)
        off_mag = off_mag.squeeze(-1)                            # (B,1,1,1)
        off_gate = torch.sigmoid(5.0 * off_mag)                  # scale factor can be tuned

        g = self.gate(y) * off_gate
        out = y_id + r * g
        return out




class SCSA(nn.Module):
    """SCSA-lite: spatial multi-semantic + channel self-attn (lightweight).
    Use it as a plug-in attention after feature fusion.
    """
    def __init__(self, c, heads=4, reduction=4):
        super().__init__()
        self.heads = heads
        self.dk = max(c // heads, 16)

        # spatial priors (multi-kernel depthwise conv)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.sproj = nn.Conv2d(c, c, 1, 1, 0, bias=False)

        # channel self-attn (qkv on pooled tokens)
        self.q = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(c, self.dk * heads, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(self.dk * heads, c, 1, 1, 0, bias=False)

        # gating
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // reduction, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # spatial enhancement
        s = self.dw3(x) + self.dw5(x)
        s = self.sproj(s)
        x1 = x + s

        # channel self-attn on global token
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x1)

        q = F.adaptive_avg_pool2d(q, 1).flatten(2)
        k = F.adaptive_avg_pool2d(k, 1).flatten(2)
        v = F.adaptive_avg_pool2d(v, 1).flatten(2)

        b = x.size(0)
        q = q.view(b, self.heads, self.dk, 1)
        k = k.view(b, self.heads, self.dk, 1)
        v = v.view(b, self.heads, self.dk, 1)

        attn = torch.softmax((q * k).sum(dim=2, keepdim=True), dim=-1)
        y = (attn * v).view(b, self.heads * self.dk, 1, 1)
        y = self.out(y)

        g = self.gate(x1)
        return x1 + y * g


__all__ = ["ADown", "DySample", "RFAConv", "DyRFAUp", "SCSA"]
