#!/usr/bin/env python3
"""Read a trained checkpoint: test accuracy, firing-rate marginals, learned targets, or the
membrane statistics behind the softmax figure.

    python analyze.py --ckpt runs/SHD_attn_homeo_layer_s0.ck.best --dataset SHD [arm args]
    python analyze.py ... --marginal 10
    python analyze.py ... --probe model.1.snn_attn.ProbLeaky_layer --variances 0.1,10,100,1000 --out probe/

--marginal N   rate matrix R[sample, neuron] of the hidden neurons over one test batch after N
               warm-up batches: mean rate, s.d. of the per-neuron means, s.d. of the per-sample means.
--probe NAME   per-step membrane mean, s.d., threshold and spiking probability of the named HomeoLIF
               on one test batch, for each surrogate variance, into <out>/probe_vs<variance>.npz.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homeolayer.experiment import Setup, add_arm_args, load_checkpoint


def main():
    p = add_arm_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--marginal', type=int, default=None, metavar='N')
    p.add_argument('--probe', default=None, metavar='MODULE')
    p.add_argument('--variances', default='0.1,10,100,1000')
    p.add_argument('--out', default='probe')
    p.add_argument('--max-batches', type=int, default=None, help='evaluate on the first batches only')
    a = p.parse_args()

    s = Setup(a)
    ck = load_checkpoint(s.model, a.ckpt, s.device)
    s.model.eval()
    print(f"[loaded] {a.ckpt}: epoch {ck['epoch']}, logged test accuracy {ck['acc']:.2f}%", flush=True)
    layers = s.model.neurons()
    for n, m in layers:
        if m.learn_target:
            print(f'  target {n:44s} r* = {float(m.target):.4f}')

    if a.marginal is not None:
        acc = {}
        hooks = [m.register_forward_hook(lambda mod, i, o, n=n: acc.__setitem__(
            n, o.detach().reshape(o.shape[0], -1).float() + acc.get(n, 0))) for n, m in layers]
        it = iter(s.test_loader)
        with torch.no_grad():
            for _ in range(a.marginal):
                s.forward(next(it)[0])
            acc.clear()
            s.forward(next(it)[0])
        for h in hooks:
            h.remove()
        R = torch.cat([acc[n] / s.T for n, _ in layers], dim=1)
        print(f'MARGINAL B={R.shape[0]} N={R.shape[1]} T={s.T} mean_rate={float(R.mean()):.4f} '
              f'sd_per_neuron={float(R.mean(0).std()):.4f} sd_per_sample={float(R.mean(1).std()):.4f}')
        for n, _ in layers:
            Rl = acc[n] / s.T
            print(f'  layer {n:44s} N={Rl.shape[1]:5d} mean={float(Rl.mean()):.4f} '
                  f'sd_neuron={float(Rl.mean(0).std()):.4f} sd_sample={float(Rl.mean(1).std()):.4f}')
        return

    if a.probe:
        mod = dict(s.model.named_modules())[a.probe]
        os.makedirs(a.out, exist_ok=True)
        x, _ = next(iter(s.test_loader))
        rstar = float(mod.target)
        for vs in [float(v) for v in a.variances.split(',')]:
            for _, m in layers:
                m.var_surrogate = vs
            rec = {'mu': [], 'sd': [], 'th': [], 'p': []}

            def hook(m, i, o):
                B = m.mem_avg_pre.shape[0]
                rec['mu'].append(m.mem_avg_pre.detach().reshape(B, -1).cpu().numpy())
                rec['sd'].append(m.var_pre.detach().clamp(min=0).sqrt().reshape(B, -1).cpu().numpy())
                rec['th'].append(m.thres.detach().reshape(-1).cpu().numpy())
                rec['p'].append(m.p_sp.detach().reshape(B, -1).cpu().numpy())
            h = mod.register_forward_hook(hook)
            with torch.no_grad():
                s.forward(x)
            h.remove()
            out = os.path.join(a.out, f'probe_vs{vs:g}.npz')
            np.savez(out, mu=np.stack(rec['mu']), sd=np.stack(rec['sd']), th=np.stack(rec['th']),
                     p=np.stack(rec['p']), rstar=rstar)
            print(f'PROBE variance={vs:g} layer={a.probe} T={len(rec["mu"])} r*={rstar:.4f} -> {out}')
        return

    acc, unseen = s.evaluate(a.max_batches)
    print(f'TEST accuracy={acc:.2f}%  unseen-speaker accuracy={unseen:.2f}%')


if __name__ == '__main__':
    main()
