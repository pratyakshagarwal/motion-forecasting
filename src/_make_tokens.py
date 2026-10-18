import numpy as np

def lane_chunks(z, c=8, max_chunks=224):
    lv, off = z["lane_vecs"], z["lane_offsets"]
    mid = (lv[:, :2] + lv[:, 2:4]) / 2
    keep = (mid[:, 0] > -60) & (mid[:, 0] < 120) & (np.abs(mid[:, 1]) <= 50)

    chunks, dists = [], []
    for k in range(len(off) - 1):
        sl = slice(off[k], off[k + 1])
        v = lv[sl][keep[sl]]                          # kept vectors of THIS lane only
        for s in range(0, len(v), c):
            piece = v[s:s + c]                        # 1..c vectors, one lane
            chunks.append(piece)
            ctr = ((piece[:, :2] + piece[:, 2:4]) / 2).mean(0)
            dists.append(np.hypot(ctr[0], ctr[1]))    # distance of the chunk center from the focal car

    order = np.argsort(dists)[:max_chunks]            # nearest first, then cap
    lanes = np.zeros((len(order), c, 5), np.float32)
    mask = np.zeros((len(order), c), bool)
    for i, j in enumerate(order):
        n = len(chunks[j])
        lanes[i, :n] = chunks[j]
        mask[i, :n] = True

    # assert mask.sum() == keep.sum(),  "the number of kept vectors, and lanes[~mask] should be all zeros."
    return lanes, mask

def agent_tokens(z, max_agents=48):
    pos, vel, valid, typ = z["agent_pos"], z["agent_vel"], z["agent_valid"], z["agent_type"]
    assert valid[0, 49], "focal car must be present at t=49"

    x, y = pos[:, 49, 0], pos[:, 49, 1]
    cand = valid[:, 49] & (x > -60) & (x < 120) & (np.abs(y) <= 50)     # 1. filter first

    idx = np.flatnonzero(cand)
    idx = idx[np.argsort(np.hypot(x[idx], y[idx]), kind="stable")]       # 2. then sort, nearest first
    idx = np.concatenate([[0], idx[idx != 0]])[:max_agents]              # 3. focal forced to the front, then cap

    p, v, m = pos[idx, :50], vel[idx, :50], valid[idx, :50]              # history only (steps 0..49)

    d = np.zeros_like(p)
    d[:, 1:] = p[:, 1:] - p[:, :-1]
    d[:, 1:] *= (m[:, 1:] & m[:, :-1])[..., None]                        # a step needs both ends valid

    feats = np.concatenate([p, v, d], axis=-1).astype(np.float32)        # (A, 50, 6)
    feats[~m] = 0
    assert  feats[0, -1, 0] == 0.0 , "The 0th index must be focal car at origin"
    return feats, m, typ[idx]    

def make_tokens(z, max_agents=48, max_chunks=224, c=8):
    lanes, lmask = lane_chunks(z, c=c, max_chunks=max_chunks)
    feats, amask, typ = agent_tokens(z, max_agents=max_agents)
    return {"agents": feats, "agent_m": amask, "agent_t": typ,
            "lanes": lanes, "lane_m": lmask,
            "v49": z["agent_vel"][0, 49].astype(np.float32),     # velocity, straight from the file
            "fut": z["focal_fut"].astype(np.float32)}            # this scene's future, via the parameter

if __name__ == '__main__':pass