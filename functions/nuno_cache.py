import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch


def _hash_tensor(t: torch.Tensor) -> str:
    t_cpu = t.detach().to('cpu').contiguous().view(-1).to(torch.float32)
    m = hashlib.sha1()
    m.update(t_cpu.numpy().tobytes())
    return m.hexdigest()


def make_key(points: torch.Tensor, cfg: Dict[str, Any]) -> str:
    h_points = _hash_tensor(points)
    h_cfg = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode('utf-8')).hexdigest()
    return f"{h_points}_{h_cfg}"


class NunoCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def path_for(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pt")

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        try:
            data = torch.load(path, map_location='cpu')
            return data
        except Exception:
            return None

    def save(self, key: str, data: Dict[str, Any]) -> None:
        path = self.path_for(key)
        torch.save(data, path)

