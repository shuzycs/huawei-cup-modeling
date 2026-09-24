"""公共工具：配置加载、路径解析、随机种子、JSON/日志读写。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """把配置中的相对路径解析为绝对路径（相对仓库根目录）。"""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def output_dir(cfg: Dict[str, Any]) -> Path:
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_parent(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# 随机性与确定性
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch 一定存在，这里只做兜底
        pass


def stable_seed(*parts: Any, base: int = 0) -> int:
    """与 Python 内置 hash() 无关的稳定种子（跨进程一致）。"""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:8], "big") + int(base)) % (2**31 - 1)


def np_rng_for(seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(seed, *parts))


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def save_json(path: str | os.PathLike, obj: Any) -> Path:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default)
    return p


def load_json(path: str | os.PathLike) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


class Tee:
    """同时写 stdout 与日志文件。"""

    def __init__(self, log_path: str | os.PathLike):
        self.terminal = sys.stdout
        self.log_path = ensure_parent(log_path)
        self.handle = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        self.handle.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.handle.flush()

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


def setup_logging(log_path: Optional[str | os.PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("problem2")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path is not None:
        handler = logging.FileHandler(ensure_parent(log_path), encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_size_mb(path: str | os.PathLike) -> float:
    return round(Path(path).stat().st_size / 2**20, 3)


# --------------------------------------------------------------------------------------
# 环境信息
# --------------------------------------------------------------------------------------
def environment_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": sys.platform,
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_device_capability"] = f"{props.major}.{props.minor}"
            info["cuda_device_memory_gb"] = round(props.total_memory / 2**30, 2)
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        info["transformers_error"] = repr(exc)
    try:
        info["numpy"] = np.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:
        info["nvidia_smi"] = f"unavailable: {exc!r}"
    return info


def count_parameters(module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def human_int(n: int) -> str:
    return f"{n:,}"


def fmt_float(x: float, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "nan"
    return f"{x:.{nd}f}"


def mean_std(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def list_files_sorted(directory: str | os.PathLike, pattern: str = "*.pkl") -> List[Path]:
    return sorted(resolve(directory).glob(pattern), key=lambda p: p.name)
