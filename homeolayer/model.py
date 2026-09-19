"""The attention network and the feedforward network, built from HomeoLayer blocks.

Attention layer, per timestep, on the spikes s of the previous block:
    q, k, v = W_q s, W_k s, W_v s
    A      <- A + eta (v~ - A k~) k~^T      k~, v~ synaptic traces of k and v; no gradient
    y       = g c A q / ||A||_F             per head, c = 5.2 sqrt(d), g learned
    out     = HomeoLayer(HomeoLIF(y)) + s
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HomeoLIF, HomeoLayer, Projection, SynapticTrace


class FastWeightAttention(nn.Module):
    def __init__(self, t_syn, t_mem, dp, hidden_size, num_heads, head_dim, attn_lr, attn_scale=None,
                 weight_normalized=True, device=None, dtype=None, **neuron):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.attn_lr = float(attn_lr)
        self.attn_scale = float(attn_scale if attn_scale is not None else 5.2 * head_dim ** 0.5)
        self.trace_gain = dp * t_syn
        width = num_heads * head_dim
        self.q_proj = Projection(hidden_size, width, weight_normalized, device, dtype)
        self.k_proj = Projection(hidden_size, width, weight_normalized, device, dtype)
        self.v_proj = Projection(hidden_size, width, weight_normalized, device, dtype)
        self._value_traces = SynapticTrace(t_syn, dp)
        self._key_traces = SynapticTrace(t_syn, dp)
        self.attn_out_gain = nn.Parameter(torch.ones((), device=device, dtype=dtype))
        self.ProbLeaky_layer = HomeoLIF(t_mem, hidden_size, device=device, dtype=dtype, **neuron)
        self.attn_mat = None

    def reset_states(self):
        self._value_traces.reset_states()
        self._key_traces.reset_states()
        self.ProbLeaky_layer.reset_states()
        self.attn_mat = None

    def forward(self, s, noise_var=0.0):
        B, H, D = s.shape[0], self.num_heads, self.head_dim
        q = self.q_proj(s).view(B, H, D)
        k = self.k_proj(s).view(B, H, D)
        v = self.v_proj(s).view(B, H, D)
        with torch.no_grad():
            if self.attn_mat is None:
                self.attn_mat = torch.zeros(B, H, D, D, device=s.device, dtype=s.dtype)
            v_tr = self._value_traces(v) * self.trace_gain
            k_tr = self._key_traces(k)
            A = self.attn_mat.detach()
            delta = v_tr.unsqueeze(-1) - A @ k_tr.unsqueeze(-1)
            self.attn_mat = A + self.attn_lr * delta @ k_tr.unsqueeze(-2)
        nrm = self.attn_mat.flatten(2).norm(dim=-1).clamp(min=1e-8).view(B, H, 1, 1)
        y = (self.attn_mat / nrm * self.attn_scale) @ q.unsqueeze(-1)
        y = y.squeeze(-1) * self.attn_out_gain
        return self.ProbLeaky_layer(y, noise_var)


class AttentionBlock(nn.Module):
    def __init__(self, t_syn, t_mem, dp, hidden_size, num_heads, head_dim, attn_lr, attn_scale=None,
                 weight_normalized=True, device=None, dtype=None, **neuron):
        super().__init__()
        self.snn_attn = FastWeightAttention(t_syn, t_mem, dp, hidden_size, num_heads, head_dim, attn_lr,
                                            attn_scale, weight_normalized, device, dtype, **neuron)
        self.out_proj = HomeoLayer(num_heads * head_dim, hidden_size, t_syn, t_mem, dp, weight_normalized,
                                   device=device, dtype=dtype, **neuron)

    def reset_states(self):
        self.snn_attn.reset_states()
        self.out_proj.reset_states()

    def forward(self, s, noise_var=0.0):
        o = self.snn_attn(s, noise_var).reshape(s.shape[0], -1)
        return self.out_proj(o, noise_var) + s


class SpikingNet(nn.Module):
    """input block -> attention blocks (num_attn_layers > 0) or feedforward blocks -> readout.
    `neuron` holds the HomeoLIF keyword arguments shared by every block."""

    def __init__(self, t_syn, t_mem, dp, input_size, hidden_size, num_heads, head_dim, num_classes,
                 num_attn_layers=2, num_ff_blocks=1, attn_lr=0.01, attn_scale=None,
                 weight_normalized=True, target_hidden=0.10, target_input=0.01,
                 input_bin_size=1, aug_channel_shift=0, device=None, dtype=None, **neuron):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_bin_size = int(input_bin_size or 1)
        self.aug_channel_shift = int(aug_channel_shift or 0)
        hidden = dict(neuron, target=target_hidden)
        blocks = [HomeoLayer(input_size, hidden_size, t_syn, t_mem, dp, weight_normalized,
                             device=device, dtype=dtype, **dict(neuron, target=target_input))]
        if num_attn_layers > 0:
            blocks += [AttentionBlock(t_syn, t_mem, dp, hidden_size, num_heads, head_dim, attn_lr, attn_scale,
                                      weight_normalized, device, dtype, **hidden)
                       for _ in range(int(num_attn_layers))]
        else:
            blocks += [HomeoLayer(hidden_size, hidden_size, t_syn, t_mem, dp, weight_normalized,
                                  device=device, dtype=dtype, **hidden)
                       for _ in range(int(num_ff_blocks))]
        blocks.append(HomeoLayer(hidden_size, self.num_classes, t_syn, t_mem, dp, weight_normalized,
                                 output_layer=True, device=device, dtype=dtype, **dict(neuron, target=target_input)))
        self.model = nn.ModuleList(blocks)

    @staticmethod
    def binned_size(input_size, bin_size):
        bin_size = int(bin_size or 1)
        return int(input_size) if bin_size <= 1 else -(-int(input_size) // bin_size)

    def reset_states(self):
        for m in self.model:
            m.reset_states()

    def neurons(self, hidden_only=True):
        return [(n, m) for n, m in self.named_modules()
                if isinstance(m, HomeoLIF) and not (hidden_only and m.output_layer)]

    def augment(self, x):
        """Random shift of the channel axis by up to +-S channels, training only."""
        if not self.training or not self.aug_channel_shift:
            return x
        shape = x.shape
        b, t = shape[0], shape[1]
        x = x.reshape(b, t, -1)
        S = self.aug_channel_shift
        sh = torch.randint(-S, S + 1, (b,), device=x.device)
        out = torch.zeros_like(x)
        for sv in sh.unique():
            idx = (sh == sv).nonzero(as_tuple=True)[0]
            k = int(sv)
            if k > 0:
                out[idx, :, k:] = x[idx][:, :, :-k]
            elif k < 0:
                out[idx, :, :k] = x[idx][:, :, -k:]
            else:
                out[idx] = x[idx]
        return out.reshape(shape)

    @staticmethod
    def blend_batch(x, y, prob=0.5):
        """Same-class blending: with probability `prob` a sample is mixed with another sample of
        its class, aligned by center of mass in time, each spike taken from either source."""
        if prob <= 0:
            return x, y
        b, t = x.shape[0], x.shape[1]
        flat = x.reshape(b, t, -1)
        e = (flat - flat.amin(dim=(1, 2), keepdim=True)).sum(-1)
        tot = e.sum(1).clamp(min=1e-8)
        com = (e * torch.arange(t, device=x.device, dtype=e.dtype)).sum(1) / tot
        out = x.clone()
        pick = torch.rand(b, device=x.device) < prob
        for i in pick.nonzero(as_tuple=True)[0].tolist():
            same = (y == y[i]).nonzero(as_tuple=True)[0]
            same = same[same != i]
            if len(same) == 0:
                continue
            j = int(same[torch.randint(len(same), (1,), device=same.device)])
            shift = int(round(float(com[i] - com[j])))
            shift = max(-(t - 1), min(t - 1, shift))
            src = torch.zeros_like(x[j])
            if shift > 0:
                src[shift:] = x[j][:t - shift]
            elif shift < 0:
                src[:shift] = x[j][-shift:]
            else:
                src = x[j]
            m = (torch.rand_like(x[i]) < 0.5).to(x.dtype)
            out[i] = x[i] * m + src * (1 - m)
        return out, y

    def bin_input(self, x):
        if self.input_bin_size <= 1:
            return x
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b, t, -1)
        pad = (-x.shape[-1]) % self.input_bin_size
        if pad:
            x = F.pad(x, (0, pad))
        return x.reshape(b, t, -1, self.input_bin_size).sum(-1)

    def forward(self, x, tmax=None, t0=0, noise_var=0.0):
        """x: [B, T, ...] frames. Returns the readout membrane averaged over the steps after t0."""
        self.reset_states()
        x = self.bin_input(self.augment(x))
        B, T = x.shape[0], x.shape[1]
        steps = T if tmax is None else min(T, tmax)
        out = torch.zeros(B, self.num_classes, device=x.device, dtype=x.dtype)
        for t in range(steps):
            h = x[:, t].reshape(B, -1)
            for m in self.model:
                h = m(h, noise_var)
            out = out + h
        return out / (steps - t0)
