# data.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

def _transform(origin, theta, pos, vel, valid):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s,  c]])                  

    pos_l = np.zeros_like(pos)
    pos_l[valid] = (pos[valid] - origin) @ R 

    vel_l = np.zeros_like(vel)
    vel_l[valid] = vel[valid] @ R         
    return pos_l, vel_l

def lane_to_vectors(lane):
    pts = np.array([[p["x"], p["y"]] for p in lane["centerline"]])      # (n, 2)
    flag = np.full((len(pts) - 1, 1), float(lane["is_intersection"]))   # (n-1, 1)
    return np.concatenate([pts[:-1], pts[1:], flag], axis=1)   

def build_scenario(root, sd):

    TYPE_IDS = {"vehicle": 0, "pedestrian": 1, "motorcyclist": 2, "riderless_bicycle": 3,
            "construction": 4, "static": 5, "background": 6, "bus":8, "unknown": 7, "cyclist":9}

    
    d = pd.read_parquet(next((root/sd).glob("*.parquet")))
    focal_id = d.focal_track_id.iloc[0]

    # agents that have any observed row, focal first, rest sorted
    ids = d[d.observed].track_id.unique().tolist()
    ids = [focal_id] + sorted(i for i in ids if i != focal_id)
    index_of = {tid: k for k, tid in enumerate(ids)}       # track_id -> row in the array

    N = len(ids)
    pos   = np.zeros((N, 110, 2))
    vel   = np.zeros((N, 110, 2))
    valid = np.zeros((N, 110), dtype=bool)

    d = d[d.track_id.isin(index_of)]                        # drop tracks with no observed rows
    a = d.track_id.map(index_of).to_numpy()                 # agent index for every row
    t = d.timestep.to_numpy()                               # time slot for every row

    for _, row in d.iterrows():
        k, t = index_of[row.track_id], int(row.timestep)
        pos[k, t] = row.position_x, row.position_y
        vel[k, t] = row.velocity_x, row.velocity_y
        valid[k, t] = True

    f = d[d.track_id == focal_id].sort_values("timestep")
    assert len(f) == 110
    r = f[f.timestep == 49].iloc[0]
    origin = np.array([r.position_x, r.position_y])
    theta = r.heading

    pos_l, vel_l = _transform(origin, theta, pos, vel, valid)
    focal_hist, focal_fut = pos_l[0, :50], pos_l[0, 50:]

    types = d.groupby("track_id").object_type.first()   
    agent_type = np.array([TYPE_IDS[types[tid]] for tid in ids])      # KeyError names the unseen type

    mp = json.load(open(next((root/sd).glob("log_map_archive_*.json"))))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    chunks, offsets = [], [0]
    for lane in mp["lane_segments"].values():
        v = lane_to_vectors(lane)                    # (n, 5): x0, y0, x1, y1, is_intersection
        start = (v[:, 0:2] - origin) @ R
        end   = (v[:, 2:4] - origin) @ R
        chunks.append(np.concatenate([start, end, v[:, 4:]], axis=1))
        offsets.append(offsets[-1] + len(v))

    lane_vecs = np.concatenate(chunks)               # (sum of n_i, 5)
    lane_offsets = np.array(offsets)                 # (n_lanes + 1,)

        # (after lane_vecs / lane_offsets)

    out = dict(scenario_id=sd,
               origin=origin, theta=np.float64(theta),
               focal_hist=focal_hist, focal_fut=focal_fut,
               agent_pos=pos_l, agent_vel=vel_l, agent_valid=valid, agent_type=agent_type,
               lane_vecs=lane_vecs, lane_offsets=lane_offsets)

    # --- asserts ---
    assert np.allclose(focal_hist[-1], 0, atol=1e-6)
    assert np.allclose(pos_l[0, :50], focal_hist) and np.allclose(pos_l[0, 50:], focal_fut)
    assert valid[0].all()
    assert (pos_l[~valid] == 0).all()
    assert focal_hist.shape == (50, 2) and focal_fut.shape == (60, 2)
    assert lane_offsets[0] == 0 and lane_offsets[-1] == len(lane_vecs)
    assert (np.diff(lane_offsets) >= 1).all()
    for k in ("focal_hist", "focal_fut", "agent_pos", "agent_vel", "lane_vecs"):
        assert np.isfinite(out[k]).all(), k

    # independent lane check: raw point -> explicit formula (not the matrix)
    p0 = mp["lane_segments"][next(iter(mp["lane_segments"]))]["centerline"][0]
    dx, dy = p0["x"] - origin[0], p0["y"] - origin[1]
    expected = np.array([np.cos(theta)*dx + np.sin(theta)*dy,
                         -np.sin(theta)*dx + np.cos(theta)*dy])
    assert np.allclose(lane_vecs[0, :2], expected)

    return out

def save_scenario(out, out_dir):
    keep64 = {"origin", "theta"}
    arrays = {}
    for k, v in out.items():
        v = np.asarray(v)                                   # str -> 0-d array, floats stay floats
        if v.dtype == np.float64 and k not in keep64:
            v = v.astype(np.float32)
        arrays[k] = v
    np.savez_compressed(out_dir / f"{out['scenario_id']}.npz", **arrays)

if __name__ == '__main__':pass