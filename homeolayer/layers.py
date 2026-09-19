"""HomeoLayer: synaptic trace -> L2-row-normalized linear map -> per-feature affine -> HomeoLIF.

HomeoLIF is a leaky integrate-and-fire neuron with a hard reset whose threshold is moved once per
timestep by theta <- theta + eta_theta * phi_bar * (r_bar - r*), where r_bar is the mean spike count
and phi_bar the mean one-bit flip statistic over a window of T_w steps, pooled over the layer's
neurons ('layer', one threshold per sample) or over the batch ('batch', one per neuron). The rate
target r* = sigmoid(rho) is a trained scalar. The weight gradient is the closed-form density of the
membrane at the threshold, propagated through the neuron's first two moments.

The same classes express the standard baselines (pool='none', a boxcar surrogate, unnormalized
weights, optional BNTT or LayerNorm).
"""
import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class L2Rows(nn.Module):
    def forward(self, w):
        return F.normalize(w, p=2, dim=1)


class SynapticTrace(nn.Module):
    """c <- c exp(-1/tau_syn) + p0 x, returned as c / (p0 tau_syn)."""

    def __init__(self, t_syn, dp):
        super().__init__()
        self.t_syn, self.dp = t_syn, dp
        self.decay = math.exp(-1 / t_syn)
        self.trace = None

    def reset_states(self):
        self.trace = None

    def forward(self, x):
        if self.trace is None:
            self.trace = torch.zeros_like(x)
        self.trace = self.trace.detach() * self.decay + self.dp * x
        return self.trace / (self.dp * self.t_syn)


class SpikeFn(torch.autograd.Function):
    """Heaviside spike with hard reset; the backward scales the gradient by `grad_scale` and
    returns its negative for the threshold."""

    @staticmethod
    def forward(ctx, mem, thres, grad_scale):
        ctx.save_for_backward(grad_scale)
        spike = (mem >= thres).to(mem.dtype)
        return spike, mem * (1 - spike)

    @staticmethod
    def backward(ctx, grad_spike, _grad_mem):
        (scale,) = ctx.saved_tensors
        g = grad_spike * scale
        return g, -g, None


def normal_pdf(x, mu, var):
    return torch.exp(-0.5 * (x - mu) ** 2 / var) / (2.0 * math.pi * var) ** 0.5


def spiking_probability(mu, thres, var):
    return 0.5 * (1 - torch.erf((thres - mu) / (var ** 0.5 * 2 ** 0.5)))


class HomeoLIF(nn.Module):
    """LIF neuron with the homeostatic threshold controller (pool 'layer' or 'batch') or a fixed
    threshold (pool 'none'). `output_layer` makes it a non-spiking leaky integrator."""

    def __init__(self, t_mem, n_units, thres_init=1.0, thres_window=1, target=0.1, lr_thres=10.0,
                 pool='layer', learn_target=False, affine=False, norm=None, norm_steps=250,
                 var_surrogate=0.1, surrogate=None, surrogate_width=1.0, one_bit_grad=False,
                 output_layer=False, device=None, dtype=None):
        super().__init__()
        assert pool in ('layer', 'batch', 'none'), pool
        assert norm in (None, 'bntt', 'ln'), norm
        assert surrogate in (None, 'boxcar'), surrogate
        self.t_mem = t_mem
        self.alpha = math.exp(-1 / t_mem)
        self.n_units = int(n_units)
        self.thres_init = float(thres_init)
        self.thres_window = int(thres_window)
        self.p_mean_desired = float(target)
        self.lr_thres = float(lr_thres)
        self.pool = pool
        self.output_layer = bool(output_layer)
        self.var_surrogate = float(var_surrogate)
        self.surrogate, self.surrogate_width = surrogate, float(surrogate_width)
        self.one_bit_grad = bool(one_bit_grad)
        self.learn_target = bool(learn_target) and not self.output_layer
        if self.learn_target:
            p0 = min(max(self.p_mean_desired, 1e-4), 1 - 1e-4)
            self._target_raw = nn.Parameter(torch.tensor(math.log(p0 / (1 - p0)), device=device, dtype=dtype))
        self.affine = bool(affine) and not self.output_layer
        if self.affine:
            self.aff_w = nn.Parameter(torch.ones(self.n_units, device=device, dtype=dtype))
            self.aff_b = nn.Parameter(torch.zeros(self.n_units, device=device, dtype=dtype))
        self.norm = None
        self.norm_per_t = False
        if norm is not None and not self.output_layer:
            if norm == 'bntt':
                self.norm = nn.ModuleList([nn.BatchNorm1d(self.n_units, device=device, dtype=dtype)
                                           for _ in range(int(norm_steps))])
                self.norm_per_t = True
            else:
                self.norm = nn.LayerNorm(self.n_units, device=device, dtype=dtype)
        self.reset_states()

    @property
    def target(self):
        return torch.sigmoid(self._target_raw) if self.learn_target else self.p_mean_desired

    def reset_states(self):
        self.mem = None
        self.spike = None
        self.spike_prev = None
        self.thres = None
        self.mem_avg_post = None
        self.var_post = None
        self.mem_avg_pre = None
        self.var_pre = None
        self.p_sp = None
        self.phi = None
        self._t = 0
        self._phi_buf = collections.deque(maxlen=self.thres_window)
        self._rate_buf = collections.deque(maxlen=self.thres_window)

    def _pool(self, v):
        if self.pool == 'batch':
            return v.mean(dim=0)
        return v.reshape(v.shape[0], -1).mean(dim=1)

    def forward(self, x, noise_var=0.0):
        shape = x.shape
        if self.affine:
            x = (x.reshape(shape[0], -1) * self.aff_w + self.aff_b).reshape(shape)
        if self.norm is not None:
            nrm = self.norm[min(self._t, len(self.norm) - 1)] if self.norm_per_t else self.norm
            x = nrm(x.reshape(shape[0], -1)).reshape(shape)
            self._t += 1
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        if self.output_layer:
            self.mem = self.alpha * self.mem.detach() + x
            return self.mem

        if self.thres is None:
            tshape = x.shape[1:] if self.pool == 'batch' else (x.shape[0],)
            self.thres = torch.full(tshape, self.thres_init, device=x.device, dtype=x.dtype)
        if self.spike_prev is None:
            self.spike_prev = torch.zeros_like(x)
        thres = (self.thres.unsqueeze(0) if self.pool == 'batch'
                 else self.thres.view(-1, *[1] * (x.dim() - 1))).expand_as(x)
        self.mem = self.alpha * self.mem.detach() + x

        if self.one_bit_grad:
            grad_scale = torch.zeros_like(self.mem) if self.phi is None else self.phi
        else:
            if self.mem_avg_post is None:
                self.mem_avg_post = torch.zeros_like(self.mem)
                self.var_post = torch.zeros_like(self.mem)
            self.mem_avg_pre = self.alpha * self.mem_avg_post.detach() + x
            self.var_pre = self.alpha ** 2 * self.var_post.detach() + self.var_surrogate
            p_sp = spiking_probability(self.mem_avg_pre, thres, self.var_pre)
            self.p_sp = p_sp
            self.phi = normal_pdf(thres, self.mem_avg_pre, self.var_pre)
            grad_scale = self.phi.detach()
            self.mem_avg_post = (1 - p_sp) * self.mem_avg_pre.detach()
            self.var_post = ((1 - p_sp) * self.var_pre.detach()
                             + p_sp * (1 - p_sp) * self.mem_avg_pre.detach() ** 2)
        if self.surrogate == 'boxcar':
            d = (self.mem - thres).detach()
            w = self.surrogate_width
            grad_scale = (d.abs() < w).to(self.mem.dtype) / (2.0 * w)

        mem = self.mem
        if noise_var:
            mem = mem + torch.randn_like(mem) * noise_var ** 0.5
        self.spike, self.mem = SpikeFn.apply(mem, thres, grad_scale)

        s0, s1 = self.spike_prev.detach(), self.spike.detach()
        flip = 0.5 * (s0 * (1 - s1) + (1 - s0) * s1)
        if self.one_bit_grad:
            self.phi = flip
            self._phi_buf.append(flip)
        else:
            self._phi_buf.append(self._pool(flip))
        self._rate_buf.append(self._pool(s1))
        self.spike_prev = s1

        if self.pool != 'none':
            r_bar = torch.stack(list(self._rate_buf), dim=0).mean(dim=0)
            phi_bar = torch.stack(list(self._phi_buf), dim=0).mean(dim=0)
            if self.one_bit_grad:
                phi_bar = phi_bar.reshape(-1).mean(dim=0)
            self.thres = self.thres.detach() + self.lr_thres * phi_bar * (r_bar - self.target)
        return self.spike


def _init_linear(lin, weight_normalized):
    if weight_normalized:
        nn.init.xavier_uniform_(lin.parametrizations.weight.original)
    else:
        nn.init.xavier_uniform_(lin.weight.data)


class HomeoLayer(nn.Module):
    """synaptic trace -> linear map (no bias) -> HomeoLIF."""

    def __init__(self, in_features, out_features, t_syn, t_mem, dp, weight_normalized=True,
                 output_layer=False, device=None, dtype=None, **neuron):
        super().__init__()
        lin = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        if weight_normalized:
            parametrize.register_parametrization(lin, 'weight', L2Rows())
        _init_linear(lin, weight_normalized)
        self.linear_layer = nn.Sequential(
            SynapticTrace(t_syn, dp), lin,
            HomeoLIF(t_mem, out_features, output_layer=output_layer, device=device, dtype=dtype, **neuron))

    @property
    def neuron(self):
        return self.linear_layer[2]

    def reset_states(self):
        for m in self.linear_layer:
            if hasattr(m, 'reset_states'):
                m.reset_states()

    def forward(self, x, noise_var=0.0):
        x = self.linear_layer[0](x)
        x = self.linear_layer[1](x)
        return self.linear_layer[2](x, noise_var)


class Projection(nn.Module):
    """A bias-free linear map on the current step's spikes (the q/k/v projections)."""

    def __init__(self, in_features, out_features, weight_normalized=True, device=None, dtype=None):
        super().__init__()
        lin = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        if weight_normalized:
            parametrize.register_parametrization(lin, 'weight', L2Rows())
        _init_linear(lin, weight_normalized)
        self.linear_layer = nn.Sequential(lin)

    def reset_states(self):
        pass

    def forward(self, x):
        return self.linear_layer(x)
