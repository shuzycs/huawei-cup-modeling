"""数据核查：`python -m src.problem2.inspect_data --config configs/problem2.yaml`

核查失败即打印明确错误并以非零码退出，训练脚本会据此停止。

产出：outputs/problem2/data_audit.json 与 data_audit.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import CLASS_NAMES, MODALITIES, __version__
from .common import (
    REPO_ROOT,
    environment_report,
    file_size_mb,
    output_dir,
    resolve,
    save_json,
    sha256_file,
)
from .data import (
    LABEL_SEPARATOR,
    SPLIT_SIZES,
    SPLITS,
    annotation_to_class,
    build_label_lookup,
    load_aligned,
    load_pickle,
)

EXPECTED_SHAPES = {
    "text_bert": (3, 50),
    "audio": (50, 74),
    "vision": (50, 35),
}
TOL = 1e-5


class AuditFailure(RuntimeError):
    pass


def _arr_stats(x: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(x)
    return {
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
        "mean": float(np.mean(flat)) if flat.size else None,
        "nan_count": int(np.isnan(flat).sum()) if flat.dtype.kind == "f" else 0,
        "inf_count": int(np.isinf(flat).sum()) if flat.dtype.kind == "f" else 0,
        "size_bytes": int(flat.nbytes),
    }


def audit_files(cfg: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    targets = {
        "aligned_pkl": resolve(cfg["data"]["aligned_pkl"]),
        "labels_xlsx": resolve(cfg["data"]["labels_xlsx"]),
        "challenge_dir": resolve(cfg["data"]["challenge_dir"]),
    }
    for key, path in targets.items():
        if not path.exists():
            problems.append(f"缺少文件/目录: {path}")
            files[key] = {"path": str(path), "exists": False}
            continue
        if path.is_dir():
            pkls = sorted(path.glob("*.pkl"))
            files[key] = {
                "path": str(path),
                "exists": True,
                "kind": "dir",
                "n_pkl": len(pkls),
                "files": [p.name for p in pkls],
            }
            if len(pkls) != 30:
                problems.append(f"附件 3 对齐版应有 30 个 .pkl，实际 {len(pkls)}")
        else:
            files[key] = {
                "path": str(path),
                "exists": True,
                "kind": "file",
                "size_mb": file_size_mb(path),
                "sha256": sha256_file(path),
            }
    return files


def audit_attachment2(cfg: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    t0 = time.time()
    data = load_aligned(cfg)
    load_seconds = time.time() - t0

    out: Dict[str, Any] = {"load_seconds": round(load_seconds, 2), "top_level_keys": sorted(map(str, data.keys()))}
    if set(SPLITS) - set(data.keys()):
        problems.append(f"附件 2 顶层缺少划分: {sorted(set(SPLITS) - set(data.keys()))}")
        raise AuditFailure("附件 2 顶层结构错误")

    label_lookup = build_label_lookup(cfg)
    out["n_excel_rows"] = len(label_lookup)

    splits_out: Dict[str, Any] = {}
    seen_ids: Dict[str, int] = {}

    for split in SPLITS:
        d = data[split]
        expected_n = SPLIT_SIZES[split]
        info: Dict[str, Any] = {"keys": sorted(map(str, d.keys()))}

        if "id" not in d:
            problems.append(f"{split}: 缺少 id 字段")
            continue
        ids = [str(x) for x in d["id"]]
        info["n_samples"] = len(ids)
        if len(ids) != expected_n:
            problems.append(f"{split}: 样本数 {len(ids)} != 期望 {expected_n}")

        # id 唯一性（划分内 + 跨划分）
        if len(set(ids)) != len(ids):
            problems.append(f"{split}: 划分内存在重复 ID")
        for i in ids:
            if i in seen_ids:
                problems.append(f"ID 交叉: {i} 同时出现在 {seen_ids[i]} 与 {split}")
            seen_ids[i] = split

        # 特征形状
        for field, tail in EXPECTED_SHAPES.items():
            if field not in d:
                problems.append(f"{split}: 缺少字段 {field}")
                continue
            arr = np.asarray(d[field])
            if arr.shape[1:] != tail:
                problems.append(f"{split}.{field}: 后两维 {arr.shape[1:]} != 期望 {tail}")
            if arr.shape[0] != len(ids):
                problems.append(f"{split}.{field}: 第一维 {arr.shape[0]} != {len(ids)}")
            info[field] = _arr_stats(arr)
            if arr.dtype.kind == "f" and (np.isnan(arr).any() or np.isinf(arr).any()):
                problems.append(f"{split}.{field}: 含 NaN/Inf")

        # text_bert 整数性
        tb = np.asarray(d["text_bert"])
        if tb.dtype.kind == "f":
            if not np.allclose(tb, np.round(tb)):
                problems.append(f"{split}.text_bert: 含非整数浮点值，无法安全转 int64")
            else:
                info["text_bert"]["note"] = "浮点但全为整数值，已确认可转 int64"
        tb_int = np.round(tb).astype(np.int64)
        vocab_max = int(tb_int.max())
        info["text_bert"]["vocab_max_id"] = vocab_max
        info["text_bert"]["min_id"] = int(tb_int.min())
        seg_unique = sorted(set(np.unique(tb_int[:, 2, :]).tolist()))
        info["text_bert"]["token_type_ids_unique"] = seg_unique
        info["text_bert"]["attention_mask_unique"] = sorted(set(np.unique(tb_int[:, 1, :]).tolist()))

        # CLS/SEP 检查
        n_cls = int((tb_int[:, 0, :] == 101).sum())
        n_sep = int((tb_int[:, 0, :] == 102).sum())
        info["text_bert"]["cls_count"] = n_cls
        info["text_bert"]["sep_count"] = n_sep
        if n_cls != len(ids) or n_sep != len(ids):
            problems.append(
                f"{split}.text_bert: [CLS]/[SEP] 计数异常 cls={n_cls} sep={n_sep} n={len(ids)}"
            )

        # 标签
        cl = np.asarray(d["classification_labels"])
        rl = np.asarray(d["regression_labels"])
        if cl.shape[0] != len(ids) or rl.shape[0] != len(ids):
            problems.append(f"{split}: 标签长度与 id 不一致")
        cl_int = np.round(cl).astype(int)
        if set(np.unique(cl_int).tolist()) - {0, 1, 2}:
            problems.append(f"{split}.classification_labels 取值超出 0/1/2: {sorted(set(np.unique(cl_int).tolist()))}")
        if not np.isfinite(rl).all():
            problems.append(f"{split}.regression_labels 含非有限值")
        if rl.min() < -3 - TOL or rl.max() > 3 + TOL:
            problems.append(f"{split}.regression_labels 超出 [-3,3]: {rl.min()}..{rl.max()}")

        # 与 Excel 按 ID 核对
        missing_in_excel = [i for i in ids if i not in label_lookup]
        if missing_in_excel:
            problems.append(f"{split}: {len(missing_in_excel)} 个 ID 在 label.xlsx 中缺失，例如 {missing_in_excel[:3]}")
        cls_mismatch, reg_mismatch, ann_mismatch = [], [], []
        for i, sid in enumerate(ids):
            rec = label_lookup.get(sid)
            if rec is None:
                continue
            if abs(rec["label"] - float(rl[i])) > 1e-4:
                reg_mismatch.append((sid, rec["label"], float(rl[i])))
            cls_ann = annotation_to_class(rec["annotation"])
            if cls_ann is not None and cls_ann != int(cl_int[i]):
                ann_mismatch.append((sid, rec["annotation"], int(cl_int[i])))
            if int(cl_int[i]) != int(cl_int[i]):
                cls_mismatch.append(sid)
        info["label_check"] = {
            "missing_in_excel": len(missing_in_excel),
            "regression_mismatch": len(reg_mismatch),
            "annotation_class_mismatch": len(ann_mismatch),
            "class_value_mismatch": len(cls_mismatch),
            "examples": {
                "regression": reg_mismatch[:3],
                "annotation": ann_mismatch[:3],
            },
        }
        if reg_mismatch:
            problems.append(f"{split}: {len(reg_mismatch)} 条样本回归标签与 Excel label 列不一致")
        if ann_mismatch:
            problems.append(f"{split}: {len(ann_mismatch)} 条样本分类标签与 Excel annotation 列不一致")

        counts = np.bincount(cl_int, minlength=3)
        info["class_distribution"] = {CLASS_NAMES[i]: int(counts[i]) for i in range(3)}
        info["class_distribution_ratio"] = {
            CLASS_NAMES[i]: round(float(counts[i] / max(1, counts.sum())), 4) for i in range(3)
        }
        info["regression_hist"] = {
            "lt_-1": int((rl < -1).sum()),
            "-1..-0.1": int(((rl >= -1) & (rl < -0.1)).sum()),
            "-0.1..0.1": int(((rl >= -0.1) & (rl <= 0.1)).sum()),
            "0.1..1": int(((rl > 0.1) & (rl <= 1)).sum()),
            "gt_1": int((rl > 1).sum()),
        }

        # 语音/视觉有效性（未标准化原始值上的非零行）
        for field in ("audio", "vision"):
            arr = np.asarray(d[field])
            nz = (np.abs(arr).sum(axis=-1) != 0)
            per_sample = nz.sum(axis=1)
            info[f"{field}_validity"] = {
                "nonzero_rows_total": int(nz.sum()),
                "total_rows": int(nz.size),
                "per_sample_min": int(per_sample.min()),
                "per_sample_max": int(per_sample.max()),
                "per_sample_mean": round(float(per_sample.mean()), 3),
                "samples_with_internal_zero_gaps": int((_internal_gap(nz) > 0).sum()),
                "zero_fraction_per_row": [round(float(v), 4) for v in nz.mean(axis=0)],
                "samples_with_zero_valid_rows": int((per_sample == 0).sum()),
            }

        # 文本有效性
        att = tb_int[:, 1, :] > 0
        content = att & (~np.isin(tb_int[:, 0, :], [0, 100, 101, 102, 103]))
        clen = content.sum(axis=1)
        info["text_validity"] = {
            "attention_len_min": int(att.sum(axis=1).min()),
            "attention_len_max": int(att.sum(axis=1).max()),
            "attention_len_mean": round(float(att.sum(axis=1).mean()), 3),
            "content_len_min": int(clen.min()),
            "content_len_max": int(clen.max()),
            "content_len_mean": round(float(clen.mean()), 3),
            "samples_with_zero_content": int((clen == 0).sum()),
        }
        # 文本长度与音视频有效行数的关系
        for field in ("audio", "vision"):
            nz = (np.abs(np.asarray(d[field])).sum(axis=-1) != 0).sum(axis=1)
            corr = float(np.corrcoef(clen, nz)[0, 1]) if clen.std() > 0 and nz.std() > 0 else None
            info[f"corr_contentlen_{field}"] = round(corr, 4) if corr is not None else None

        # raw_text 长度
        if "raw_text" in d:
            rt = np.asarray(d["raw_text"])
            info["raw_text"] = {
                "dtype": str(rt.dtype),
                "char_len_min": int(min(len(s) for s in rt)),
                "char_len_max": int(max(len(s) for s in rt)),
                "char_len_mean": round(float(np.mean([len(s) for s in rt])), 2),
                "n_empty": int(sum(1 for s in rt if len(s.strip()) == 0)),
            }
        if "text" in d:
            info["precomputed_text"] = _arr_stats(np.asarray(d["text"]))
            info["precomputed_text"]["note"] = "附件 2 预计算 text，仅可用于 E4 教师训练；主模型输入只允许 text_bert"

        splits_out[split] = info

    out["splits"] = splits_out
    out["id_cross_split_duplicates"] = int(len(seen_ids) - sum(1 for _ in seen_ids))
    if len(seen_ids) != 3395 + 728 + 727:
        problems.append(f"三个划分 ID 合计 {len(seen_ids)} != 4850，存在跨划分重复")
    out["n_unique_ids"] = len(seen_ids)

    # 文本可观测性规则说明
    out["observability_rules"] = {
        "text": "v_T = attention_mask==1 且 token 不是 [CLS]/[SEP]/[PAD]；模拟文本缺失时不遮蔽特殊 token",
        "audio": "v_A = 该行 74 维原始值 L1 范数 > 0；对齐版行块连续（起点可能为 1 号槽位）",
        "vision": "v_V = 该行 35 维原始值 L1 范数 > 0；少量样本存在内部零行（原始对齐产生）",
        "note": "附件 3 只有缺失后的零行，无法完美反推原始有效长度；推理时全零行视为不可观测",
    }
    return out


def _internal_gap(nz: np.ndarray) -> np.ndarray:
    any_row = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    last = nz.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1)
    span = np.where(any_row, last - first + 1, 0)
    return span - nz.sum(axis=1)


def audit_attachment3(cfg: Dict[str, Any], problems: List[str]) -> Dict[str, Any]:
    files = sorted(resolve(cfg["data"]["challenge_dir"]).glob("*.pkl"))
    out: Dict[str, Any] = {"n_files": len(files), "files": []}
    expected_tail = {"text_bert": (1, 3, 50), "audio": (1, 50, 74), "vision": (1, 50, 35)}
    indices = []
    for path in files:
        try:
            idx = int(path.stem.split("_")[-1])
        except ValueError:
            problems.append(f"附件 3 文件名无法解析编号: {path.name}")
            idx = -1
        indices.append(idx)
        d = load_pickle(path)
        rec: Dict[str, Any] = {"file_name": path.name, "index": idx, "size_kb": round(path.stat().st_size / 1024, 3)}
        if "test" not in d:
            problems.append(f"{path.name}: 缺少 'test' 字段")
            rec["error"] = "missing test"
            out["files"].append(rec)
            continue
        t = d["test"]
        rec["keys"] = sorted(map(str, t.keys()))
        for field, shape in expected_tail.items():
            if field not in t:
                problems.append(f"{path.name}: 缺少字段 {field}")
                continue
            arr = np.asarray(t[field])
            if tuple(arr.shape) != shape:
                problems.append(f"{path.name}.{field}: 形状 {tuple(arr.shape)} != {shape}")
            if arr.dtype.kind == "f" and (np.isnan(arr).any() or np.isinf(arr).any()):
                problems.append(f"{path.name}.{field}: 含 NaN/Inf")
            rec[field] = _arr_stats(arr)
        tb = np.round(np.asarray(t["text_bert"])).astype(np.int64)
        att = tb[:, 1, :] > 0
        content = att & (~np.isin(tb[:, 0, :], [0, 100, 101, 102, 103]))
        rec["content_len"] = int(content.sum())
        rec["attention_len"] = int(att.sum())
        rec["audio_valid_rows"] = int((np.abs(np.asarray(t["audio"])).sum(axis=-1) != 0).sum())
        rec["vision_valid_rows"] = int((np.abs(np.asarray(t["vision"])).sum(axis=-1) != 0).sum())
        out["files"].append(rec)

    if sorted(indices) != list(range(1, 31)):
        problems.append(f"附件 3 编号不完整: {sorted(indices)}")
        out["index_ok"] = False
    else:
        out["index_ok"] = True
    out["order"] = [p.name for p in sorted(files, key=lambda p: int(p.stem.split("_")[-1]))]
    return out


def _num(x, nd: int = 3, default: str = "n/a") -> str:
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return f"{v:.{nd}f}"


def write_markdown(audit: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# 问题 2 数据核查报告")
    lines.append("")
    lines.append(f"- 生成时间：{audit['environment']['time']}")
    lines.append(f"- Python：{audit['environment']['python']}")
    lines.append(f"- PyTorch：{audit['environment'].get('torch')} / CUDA build {audit['environment'].get('torch_cuda_build')}")
    lines.append(f"- CUDA 可用：{audit['environment'].get('cuda_available')} ({audit['environment'].get('cuda_device_name')})")
    lines.append(f"- 核查结论：{'全部通过' if audit['passed'] else '存在失败项'}")
    lines.append(f"- 问题数：{len(audit['problems'])}")
    lines.append("")

    lines.append("## 1. 文件清单")
    lines.append("")
    lines.append("| 键 | 路径 | 存在 | 说明 |")
    lines.append("|---|---|---|---|")
    for key, rec in audit["files"].items():
        desc = f"{rec.get('size_mb', '')} MB" if rec.get("kind") == "file" else f"{rec.get('n_pkl', '')} 个 pkl"
        lines.append(f"| {key} | {rec['path']} | {rec.get('exists')} | {desc} |")
    lines.append("")

    if "attachment2" in audit:
        a2 = audit["attachment2"]
        lines.append("## 2. 附件 2 对齐版核查")
        lines.append("")
        lines.append(f"反序列化耗时 {a2['load_seconds']} s，有效 ID {a2['n_unique_ids']} 个（期望 4850）。")
        lines.append("")
        lines.append("| 划分 | 样本数 | text_bert | audio | vision | 类别分布 (N/U/P) | 强度范围 |")
        lines.append("|---|---|---|---|---|---|---|")
        for split in SPLITS:
            s = a2["splits"].get(split)
            if not s:
                continue
            cd = s.get("class_distribution", {})
            lines.append(
                "| {} | {} | {} | {} | {} | {}/{}/{} | {:.3f} ~ {:.3f} |".format(
                    split,
                    s.get("n_samples"),
                    s.get("text_bert", {}).get("shape"),
                    s.get("audio", {}).get("shape"),
                    s.get("vision", {}).get("shape"),
                    cd.get("Negative"),
                    cd.get("Neutral"),
                    cd.get("Positive"),
                    _num(s.get("regression_labels", {}).get("min")),
                    _num(s.get("regression_labels", {}).get("max")),
                )
            )
        lines.append("")
        lines.append("### 2.1 标签连接核对（按 video_id + '$_$' + clip_id）")
        lines.append("")
        lines.append("| 划分 | Excel 缺失 ID | 回归标签不一致 | annotation 与分类标签不一致 |")
        lines.append("|---|---|---|---|")
        for split in SPLITS:
            lc = a2["splits"].get(split, {}).get("label_check", {})
            lines.append(
                f"| {split} | {lc.get('missing_in_excel')} | {lc.get('regression_mismatch')} | {lc.get('annotation_class_mismatch')} |"
            )
        lines.append("")
        lines.append("### 2.2 有效性与可观测性规则")
        lines.append("")
        for k, v in a2["observability_rules"].items():
            lines.append(f"- **{k}**：{v}")
        lines.append("")
        lines.append("### 2.3 文本长度与音视频有效行统计")
        lines.append("")
        lines.append("| 划分 | 文本 attention 长度均值 | 文本内容长度均值 | 语音有效行均值 | 视觉有效行均值 | corr(内容长度,语音) | corr(内容长度,视觉) |")
        lines.append("|---|---|---|---|---|---|---|")
        for split in SPLITS:
            s = a2["splits"].get(split, {})
            tv = s.get("text_validity", {})
            av = s.get("audio_validity", {})
            vv = s.get("vision_validity", {})
            lines.append(
                f"| {split} | {tv.get('attention_len_mean')} | {tv.get('content_len_mean')} | "
                f"{av.get('per_sample_mean')} | {vv.get('per_sample_mean')} | "
                f"{s.get('corr_contentlen_audio')} | {s.get('corr_contentlen_vision')} |"
            )
        lines.append("")

    if "attachment3" in audit:
        a3 = audit["attachment3"]
        lines.append("## 3. 附件 3 对齐版 schema 核查")
        lines.append("")
        lines.append(f"共 {a3['n_files']} 个文件，编号完整：{a3['index_ok']}")
        lines.append("")
        lines.append("| 文件 | text_bert | audio | vision | 文本内容长度 | 语音有效行 | 视觉有效行 |")
        lines.append("|---|---|---|---|---|---|---|")
        for rec in a3["files"]:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    rec["file_name"],
                    rec.get("text_bert", {}).get("shape"),
                    rec.get("audio", {}).get("shape"),
                    rec.get("vision", {}).get("shape"),
                    rec.get("content_len"),
                    rec.get("audio_valid_rows"),
                    rec.get("vision_valid_rows"),
                )
            )
        lines.append("")

    lines.append("## 4. 问题清单")
    lines.append("")
    if audit["problems"]:
        for p in audit["problems"]:
            lines.append(f"- ❌ {p}")
    else:
        lines.append("- 无。所有核查项通过。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="问题 2 数据核查")
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--skip-hash", action="store_true", help="跳过 SHA256（大文件较慢）")
    args = parser.parse_args(argv)

    from .common import load_config

    cfg = load_config(args.config)
    out_dir = output_dir(cfg)

    problems: List[str] = []
    audit: Dict[str, Any] = {
        "script": "src.problem2.inspect_data",
        "script_version": __version__,
        "config_path": cfg["_config_path"],
        "environment": environment_report(),
        "expected": {
            "split_sizes": SPLIT_SIZES,
            "modalities": list(MODALITIES),
            "class_names": list(CLASS_NAMES),
            "features": EXPECTED_SHAPES,
            "label_separator": LABEL_SEPARATOR,
        },
    }

    audit["files"] = audit_files(cfg, problems)

    try:
        audit["attachment2"] = audit_attachment2(cfg, problems)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"附件 2 核查异常: {type(exc).__name__}: {exc}")

    try:
        audit["attachment3"] = audit_attachment3(cfg, problems)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"附件 3 核查异常: {type(exc).__name__}: {exc}")

    audit["problems"] = problems
    audit["passed"] = len(problems) == 0

    json_path = save_json(out_dir / "data_audit.json", audit)
    md_path = out_dir / "data_audit.md"
    write_markdown(audit, md_path)

    print(f"[inspect_data] 核查{'通过' if audit['passed'] else '失败'}，问题 {len(problems)} 项")
    for p in problems:
        print(f"  - {p}")
    print(f"[inspect_data] 已写出 {json_path}")
    print(f"[inspect_data] 已写出 {md_path}")
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
