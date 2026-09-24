"""预训练文本编码器：google/bert_uncased_L-2_H-128_A-2

- revision 必须是已核实的完整 commit SHA（首次运行时由脚本联网解析并写回配置）；
- 训练/验证/附件 3 使用完全相同的权重；
- 保存各文件的 SHA256，便于复现与体积核对。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .common import REPO_ROOT, ensure_parent, resolve, save_json, sha256_file

REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.txt")
OPTIONAL_FILES = ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json")


def resolve_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id)
    return info.sha


def is_pinned(revision: Optional[str]) -> bool:
    return bool(revision) and revision != "REPLACE_WITH_VERIFIED_COMMIT_SHA" and len(str(revision)) >= 40


def fetch_model(
    repo_id: str,
    revision: Optional[str],
    local_dir: str | os.PathLike,
    token: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    target = resolve(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    pinned = revision if is_pinned(revision) else None
    if pinned is None:
        pinned = resolve_revision(repo_id)
        print(f"[text_encoder] 未固定 revision，已解析 {repo_id} -> {pinned}")

    if force or not (target / "model.safetensors").exists():
        snapshot_download(
            repo_id=repo_id,
            revision=pinned,
            local_dir=str(target),
            allow_patterns=list(REQUIRED_FILES) + list(OPTIONAL_FILES),
            token=token,
        )
    manifest = build_manifest(repo_id, pinned, target)
    save_json(target / "manifest.json", manifest)
    return manifest


def build_manifest(repo_id: str, revision: str, local_dir: str | os.PathLike) -> Dict[str, Any]:
    target = resolve(local_dir)
    files = {}
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        p = target / name
        if not p.exists():
            continue
        files[name] = {
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    total = sum(v["size_bytes"] for v in files.values())
    return {
        "repo_id": repo_id,
        "revision": revision,
        "url": f"https://huggingface.co/{repo_id}/tree/{revision}",
        "local_dir": str(target),
        "files": files,
        "total_size_bytes": total,
        "total_size_mb": round(total / 2**20, 3),
        "note": "仅保留推理必需文件；同一权重不同格式只保留 safetensors 一份",
    }


def load_tokenizer(repo_id: str, revision: Optional[str], local_dir: str | os.PathLike, token: Optional[str] = None):
    from transformers import AutoTokenizer

    target = resolve(local_dir)
    source = str(target) if (target / "vocab.txt").exists() else repo_id
    kwargs: Dict[str, Any] = {"token": token}
    if source == repo_id and is_pinned(revision):
        kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(source, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.mask_token or "[PAD]"
    return tok


def load_bert(repo_id: str, revision: Optional[str], local_dir: str | os.PathLike, token: Optional[str] = None):
    from transformers import AutoConfig, AutoModel

    target = resolve(local_dir)
    source = str(target) if (target / "config.json").exists() else repo_id
    kwargs: Dict[str, Any] = {"token": token}
    if source == repo_id and is_pinned(revision):
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(source, **kwargs)
    model = AutoModel.from_pretrained(source, **kwargs)
    return config, model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载/固定预训练文本编码器")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--force", action="store_true", help="即使本地已存在也重新下载")
    parser.add_argument(
        "--write-config-revision",
        action="store_true",
        help="把解析到的 commit SHA 写回 configs/problem2.yaml",
    )
    args = parser.parse_args(argv)

    from .common import load_config

    cfg = load_config(args.config)
    tcfg = cfg["text_model"]
    manifest = fetch_model(
        tcfg["repo_id"],
        tcfg.get("revision"),
        tcfg["local_dir"],
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.write_config_revision:
        cfg_path = Path(cfg["_config_path"])
        text = cfg_path.read_text(encoding="utf-8")
        old = str(tcfg.get("revision"))
        if old in text:
            cfg_path.write_text(text.replace(old, manifest["revision"]), encoding="utf-8")
            print(f"[text_encoder] 已把 revision 写回 {cfg_path}")
        else:
            print(f"[text_encoder] 警告：配置中未找到 '{old}'，未写回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
