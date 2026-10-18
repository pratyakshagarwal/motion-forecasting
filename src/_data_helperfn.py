import torch
import numpy as np
from  src._make_tokens import make_tokens

class FocalDatasetv2(torch.utils.data.Dataset):
    def __init__(self, files, dev): self.files, self.dev = files, dev
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        with np.load(self.files[i]) as z:
            zd = {k: z[k] for k in z.files}
        assert np.allclose(zd["focal_hist"][-1], 0, atol=1e-5), "focal car must be at the origin"
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in make_tokens(zd).items()}


def collate(batch):
    B = len(batch)
    A = max(b["agents"].shape[0] for b in batch)
    L = max(b["lanes"].shape[0] for b in batch)

    agents  = torch.zeros(B, A, 50, 6)
    agent_m = torch.zeros(B, A, 50, dtype=torch.bool)
    agent_t = torch.zeros(B, A, dtype=torch.long)
    lanes   = torch.zeros(B, L, 8, 5)
    lane_m  = torch.zeros(B, L, 8, dtype=torch.bool)

    for i, b in enumerate(batch):
        a, l = b["agents"].shape[0], b["lanes"].shape[0]
        # TODO: copy b's arrays into row i, in the first a (or l) slots
        agents[i,:a] = b['agents']; agent_m[i,:a] = b['agent_m']
        lanes[i,:l] = b['lanes']; lane_m[i,:l] = b['lane_m']
        agent_t[i,:a] = b['agent_t']

    return {
        "agents": agents, "agent_m": agent_m, "agent_t": agent_t,
        "lanes": lanes, "lane_m": lane_m,
        "agent_tok": agent_m.any(-1),     # (B, A): is this agent real at all?
        "lane_tok": lane_m.any(-1),       # (B, L): is this chunk real at all?
        "v49": torch.stack([b["v49"] for b in batch]),
        "fut": torch.stack([b["fut"] for b in batch]),
    }


if __name__ == '__main__':pass