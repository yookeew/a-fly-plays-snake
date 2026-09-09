"""
Parity check for the GPU forward pass (model.Brain device="cuda").

Runs the batched population step -- unpack_pop / initial_state_pop / step_pop --
on CPU numpy and on the device, same params and same random observations, and
asserts the logit trajectories agree. This is the only path ES exercises.

Run on Colab (needs a GPU + torch):
    python explore/torch_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from brain import make_synthetic
from model import Brain
from snake import FEATURE_NAMES


def trajectory(brain, thetas, obs_seq):
    params = brain.unpack_pop(thetas)
    C, E = thetas.shape[0], obs_seq.shape[2]
    h = brain.initial_state_pop(C, E)
    out = []
    for obs in obs_seq:                     # (T, C, E, n_obs)
        h, logits = brain.step_pop(h, obs, params)
        out.append(np.asarray(logits))      # numpy on both paths
    return np.stack(out)


def main():
    import torch
    assert torch.cuda.is_available(), "no CUDA device"

    n_obs = len(FEATURE_NAMES)
    cx = make_synthetic(n_obs=n_obs, n_neurons=4000, n_out=32, seed=1)
    C, E, T = 6, 4, 40
    rng = np.random.default_rng(0)
    thetas = np.stack([Brain(cx).init_params(seed=s, gain_init=2.0)
                       for s in range(C)])
    obs_seq = rng.normal(0, 1, (T, C, E, n_obs)).astype(np.float32)

    lg_cpu = trajectory(Brain(cx, inner_steps=8, device="cpu"), thetas, obs_seq)
    lg_gpu = trajectory(Brain(cx, inner_steps=8, device="cuda"), thetas, obs_seq)

    err = np.abs(lg_cpu - lg_gpu)
    argmax_agree = (lg_cpu.argmax(2) == lg_gpu.argmax(2)).mean()
    print(f"logit max|err| {err.max():.2e}   mean|err| {err.mean():.2e}")
    print(f"argmax agreement {argmax_agree:.1%}")
    assert err.max() < 1e-2, "GPU path diverges from numpy"
    assert argmax_agree > 0.99, "GPU path picks different actions"
    print("PARITY OK")


if __name__ == "__main__":
    main()
