# -*- coding: utf-8 -*-
import torch
from networks import ST_Actor
from config import cfg


def test_actor():
    print(f"Testing ST_Actor with N_FRAMES={cfg.N_FRAMES} (OBS_DIM={cfg.OBS_DIM})...")
    bs = 32
    n_uav = 4
    obs = torch.randn(bs, cfg.OBS_DIM)

    neigh_feats = torch.randn(bs, n_uav - 1, cfg.HIDDEN_DIM_2)

    actor = ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM)

    my_feat = actor.extract_feat(obs)
    assert my_feat.shape == (bs, cfg.HIDDEN_DIM_2)

    mu, log_std = actor(my_feat, neigh_feats)

    assert mu.shape == (bs, cfg.ACT_DIM)
    assert log_std.shape == (bs, cfg.ACT_DIM)

    print("✅ ST_Actor Pass (Dimensions are correct)")


if __name__ == "__main__":
    test_actor()