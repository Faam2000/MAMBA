
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================
# Normalization
# ============================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# ============================
# Patch Embedding
# ============================
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================
# TRUE MAMBA SSM BLOCK
# ============================
class MambaBlockSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.d_inner = 2 * d_model
        self.dt_rank = (d_model + 15) // 16

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        A = torch.arange(1, d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        x, gate = self.in_proj(x).chunk(2, dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :x.shape[-1]]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(gate)

        return residual + self.out_proj(y)

    def ssm(self, x):
        B, L, D = x.shape
        N = self.A_log.shape[1]

        A = -torch.exp(self.A_log)
        D_skip = self.D

        x_proj = self.x_proj(x)
        delta, Bp, Cp = x_proj.split([self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB = torch.einsum('bld,bln,bld->bldn', delta, Bp, x)

        state = torch.zeros(B, D, N, device=x.device)
        ys = []

        for t in range(L):
            state = deltaA[:, t] * state + deltaB[:, t]
            y = torch.einsum('bdn,bn->bd', state, Cp[:, t])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        return y + x * D_skip


# ============================
# Light Cross Attention
# ============================
class LightCrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = RMSNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t, f):
        B, Nt, C = t.shape
        _, Nf, _ = f.shape

        t = self.norm(t)
        f = self.norm(f)

        q = self.q(t).view(B, Nt, self.heads, C // self.heads).transpose(1, 2)
        k, v = self.kv(f).chunk(2, dim=-1)
        k = k.view(B, Nf, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, Nf, self.heads, C // self.heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Nt, C)
        return self.proj(out) + t


# ============================
# FULL MODEL
# ============================
class TFMambaLite(nn.Module):
    def __init__(self, num_classes=1, embed_dim=32, depth=4, img_size=(256, 126)):
        super().__init__()

        self.t_patch = PatchEmbed(
            img_size, patch_size=(img_size[0], 16), stride=(1, 10),
            in_ch=1, embed_dim=embed_dim
        )

        self.f_patch = PatchEmbed(
            img_size, patch_size=(36, img_size[1]), stride=(20, 1),
            in_ch=1, embed_dim=embed_dim
        )

        self.t_blocks = nn.ModuleList([
            MambaBlockSSM(embed_dim) for _ in range(depth)
        ])

        self.f_blocks = nn.ModuleList([
            MambaBlockSSM(embed_dim) for _ in range(depth)
        ])

        self.cross = LightCrossAttention(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        t = self.t_patch(x)
        f = self.f_patch(x)

        for blk in self.t_blocks:
            t = blk(t)
        for blk in self.f_blocks:
            f = blk(f)

        if t.shape[1] != f.shape[1]:
            m = min(t.shape[1], f.shape[1])
            t, f = t[:, :m], f[:, :m]

        fused = self.cross(t, f)
        pooled = fused.mean(dim=1)
        return self.head(pooled)
