import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvExpert(nn.Module):
    """Lightweight convolutional expert used by IA-MoE and TD-MoE."""

    def __init__(self, indim, outdim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(indim, outdim, 1),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, 1),
        )

    def forward(self, x):
        return self.net(x)


class InteractionAdaptiveMoELayer(nn.Module):
    """Interaction-Adaptive MoE over difference and summation features."""

    def __init__(self, indim, unified_dim=196, outdim=None, n_experts=4, topk=2, noisy_gating=True):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.unified_dim = unified_dim
        self.noisy_gating = noisy_gating

        if outdim is None:
            outdim = indim

        self.shared_reduction = nn.Conv2d(indim, unified_dim, 1)
        self.experts = nn.ModuleList([ConvExpert(unified_dim, unified_dim) for _ in range(n_experts)])
        self.gate_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(unified_dim * 4, unified_dim // 2),
            nn.ReLU(),
            nn.Linear(unified_dim // 2, n_experts),
        )
        self.channel_expansion = nn.Conv2d(unified_dim, outdim, 1)
        self.identity_proj = nn.Conv2d(indim * 2, outdim, 1)

        nn.init.constant_(self.channel_expansion.weight, 0)
        nn.init.constant_(self.channel_expansion.bias, 0)

    def forward(self, x1, x2, training=True):
        b, _, h, w = x1.shape
        identity_out = self.identity_proj(torch.cat([x1, x2], dim=1))

        f1 = self.shared_reduction(x1)
        f2 = self.shared_reduction(x2)
        feat_diff = f1 - f2
        feat_sum = f1 + f2

        gate_feat = torch.cat([f1, f2], dim=1)
        gate_avg = F.adaptive_avg_pool2d(gate_feat, 1)
        gate_max = F.adaptive_max_pool2d(gate_feat, 1)
        gate_logits = self.gate_proj(torch.cat([gate_avg, gate_max], dim=1))

        if training and self.noisy_gating:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 0.1

        gate_prob = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_prob, self.topk, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        out_moe = torch.zeros((b, self.unified_dim, h, w), device=x1.device, dtype=x1.dtype)
        split_idx = self.n_experts // 2
        for i in range(b):
            for k in range(self.topk):
                idx = topk_idx[i, k]
                val = topk_vals[i, k]
                expert_input = feat_diff[i : i + 1] if idx < split_idx else feat_sum[i : i + 1]
                out_moe[i] += val * self.experts[idx](expert_input).squeeze(0)

        return identity_out + self.channel_expansion(out_moe)


class InteractionAdaptiveMoE(nn.Module):
    """Wrapper kept with the original ``MoE`` attribute name for checkpoint compatibility."""

    def __init__(self, dim, unified_dim=196, outdim=128, num_experts=4, k=2):
        super().__init__()
        self.MoE = InteractionAdaptiveMoELayer(
            indim=dim,
            unified_dim=unified_dim,
            outdim=outdim,
            n_experts=num_experts,
            topk=k,
        )

    def forward(self, x1, x2, istrain=True):
        return self.MoE(x1, x2, istrain)


class TaskDecouplingMoEHead(nn.Module):
    """Task-Decoupling MoE with two task-specific gates over a shared expert pool."""

    def __init__(self, in_dim, n_experts=4, topk=2, noisy_gating=True):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.noisy_gating = noisy_gating

        self.experts = nn.ModuleList([ConvExpert(in_dim, in_dim) for _ in range(n_experts)])
        self.gate_cd = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, n_experts),
        )
        self.gate_sem = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, n_experts),
        )

        self.res_cd = nn.Conv2d(in_dim, in_dim, 1)
        self.res_sem = nn.Conv2d(in_dim, in_dim, 1)
        self.fuse_t1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 1),
        )
        self.fuse_t2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _run_moe(self, x, gate_net, training):
        b = x.shape[0]
        gate_avg = F.adaptive_avg_pool2d(x, 1)
        gate_max = F.adaptive_max_pool2d(x, 1)
        gate_logits = gate_net(torch.cat([gate_avg, gate_max], dim=1))

        if training and self.noisy_gating:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 0.1

        gate_prob = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_prob, self.topk, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for i in range(b):
            for k in range(self.topk):
                idx = topk_idx[i, k]
                val = topk_vals[i, k]
                out[i] += val * self.experts[idx](x[i : i + 1]).squeeze(0)
        return out

    def forward(self, unified_feat, training=True):
        moe_cd = self._run_moe(unified_feat, self.gate_cd, training)
        feat_cd = self.res_cd(unified_feat) + moe_cd

        moe_sem = self._run_moe(unified_feat, self.gate_sem, training)
        feat_sem = self.res_sem(unified_feat) + moe_sem

        pair_input = torch.cat([feat_sem, feat_cd], dim=1)
        feat_t1 = self.fuse_t1(pair_input)
        feat_t2 = self.fuse_t2(pair_input)
        return feat_t1, feat_t2, feat_cd
