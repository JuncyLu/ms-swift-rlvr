#!/usr/bin/env python3
"""
ViRL39K 渐进式数据集处理脚本：下载 → 查看字段 → 转换为 GRPO 训练格式

目标格式（与 run_grpo_with_ddapo.sh / plugin 兼容）：
  - messages: [{"role": "user", "content": "<query>"}]，多模态时 content 可含 <image>
  - images: ["path1", "path2"]  # 本地绝对路径或相对 data 目录的路径
  - solution: 参考答案，用于 external_r1v_acc 等 reward（可为 \\boxed{} 或 <answer>...</answer>）

用法：
  # 仅下载并查看字段（不转换）
  python prepare_virl39k.py --step inspect --out_dir ./data/virl39k

  # 下载 + 查看 + 转换为 train.jsonl
  python prepare_virl39k.py --step all --out_dir ./data/virl39k

  # 已有本地 parquet，只做转换
  python prepare_virl39k.py --step convert --parquet ./data/virl39k/39Krelease.parquet --images_dir ./data/virl39k/images --out_dir ./data/virl39k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 确保能 import 项目内模块
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def step_download(out_dir: Path, use_hf: bool = True) -> tuple[Path | None, Path | None]:
    """Step 1: 下载 ViRL39K（用 huggingface_hub 直接下 parquet + images.zip，避免 load_dataset 的 parquet 流式解析错误）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "39Krelease.parquet"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if use_hf:
        try:
            from huggingface_hub import hf_hub_download
            print("[Step 1] 从 HuggingFace 下载 TIGER-Lab/ViRL39K（parquet + images.zip）...")
            # 直接下载 parquet 到本地，避免 datasets 库对 parquet 的流式解析导致 "Parquet magic bytes not found"
            parquet_path = Path(
                hf_hub_download(
                    repo_id="TIGER-Lab/ViRL39K",
                    filename="39Krelease.parquet",
                    repo_type="dataset",
                    local_dir=out_dir,
                )
            )
            print(f"  已保存: {parquet_path}")
            # 下载并解压 images.zip
            zip_path = hf_hub_download(
                repo_id="TIGER-Lab/ViRL39K",
                filename="images.zip",
                repo_type="dataset",
                local_dir=out_dir,
            )
            zip_path = Path(zip_path)
            if zip_path.exists():
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(images_dir)
                print(f"  已解压图片到: {images_dir}")
            return parquet_path, images_dir
        except Exception as e:
            print(f"[Step 1] 下载失败: {e}")
            print("  请手动将 39Krelease.parquet 与 images.zip 放到 --out_dir，解压 images.zip 到 out_dir/images 后使用 --step convert")
            return None, images_dir

    print("[Step 1] 未使用 HF；请将 39Krelease.parquet 和 images.zip 放到 out_dir 后运行 --step convert")
    return parquet_path if parquet_path.exists() else None, images_dir


def step_inspect(parquet_path: Path | None, images_dir: Path | None, out_dir: Path) -> None:
    """Step 2: 查看数据集字段与样例，便于确认与训练代码的兼容性。"""
    out_dir = Path(out_dir)
    if parquet_path is None or not parquet_path.exists():
        print("[Step 2] 未找到 parquet 文件，跳过 inspect。请先完成 Step 1 或指定 --parquet")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[Step 2] 需要 pandas: pip install pandas")
        return

    print("[Step 2] 查看 ViRL39K 字段与样例 ...")
    df = pd.read_parquet(parquet_path)
    columns = list(df.columns)
    print(f"  列名: {columns}")
    print(f"  行数: {len(df)}")

    # 打印第一行各字段类型与样例
    report_path = out_dir / "virl39k_schema.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ViRL39K 字段与样例\n")
        f.write("=" * 60 + "\n")
        f.write(f"列名: {columns}\n行数: {len(df)}\n\n")
        for c in columns:
            dtype = str(df[c].dtype)
            sample = df[c].iloc[0]
            if hasattr(sample, "__len__") and not isinstance(sample, str) and len(str(sample)) > 200:
                sample = str(sample)[:200] + "..."
            f.write(f"  {c} ({dtype}): {repr(sample)}\n")
        f.write("\n前 3 行 JSON 摘要:\n")
        for i in range(min(3, len(df))):
            row = df.iloc[i].to_dict()
            for k, v in row.items():
                if hasattr(v, "__len__") and not isinstance(v, str) and len(str(v)) > 150:
                    row[k] = str(v)[:150] + "..."
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"  已写入: {report_path}")

    # 检查图片列与 images 目录
    if images_dir and images_dir.exists():
        n_files = len(list(images_dir.glob("*")))
        print(f"  图片目录 {images_dir} 下文件数: {n_files}")
    image_cols = [c for c in columns if "image" in c.lower() or "img" in c.lower() or "photo" in c.lower()]
    if image_cols:
        print(f"  疑似图片列: {image_cols}")


def _infer_virl_columns(df) -> dict:
    """根据 ViRL39K 常见列名推断 question / answer / image 列。"""
    cols = set(df.columns)
    mapping = {}
    # 常见命名
    for q in ("question", "query", "prompt", "questions", "problem"):
        if q in cols:
            mapping["question"] = q
            break
    for a in ("answer", "answers", "solution", "response"):
        if a in cols:
            mapping["answer"] = a
            break
    for i in ("image", "image_path", "image_paths", "images", "img", "photo"):
        if i in cols:
            mapping["image"] = i
            break
    return mapping


def step_convert(
    parquet_path: Path,
    out_dir: Path,
    images_dir: Path | None,
    question_col: str | None = None,
    answer_col: str | None = None,
    image_col: str | None = None,
    max_samples: int | None = None,
    query_suffix: str = "",
) -> Path:
    """
    Step 3: 将 ViRL39K 转为 GRPO 所需格式并写入 train.jsonl。

    输出每条样本格式:
      {"messages": [{"role": "user", "content": "<query>"}], "images": ["path"], "solution": "<answer>"}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train.jsonl"
    images_base = Path(images_dir) if images_dir else (out_dir / "images")
    images_base.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("需要 pandas: pip install pandas")

    df = pd.read_parquet(parquet_path)
    inferred = _infer_virl_columns(df)
    qcol = question_col or inferred.get("question")
    acol = answer_col or inferred.get("answer")
    icol = image_col or inferred.get("image")

    if not qcol or qcol not in df.columns:
        raise ValueError(f"未找到 question 列，请用 --question_col 指定。当前列: {list(df.columns)}")
    if not acol or acol not in df.columns:
        raise ValueError(f"未找到 answer 列，请用 --answer_col 指定。当前列: {list(df.columns)}")

    n = len(df) if max_samples is None else min(max_samples, len(df))
    converted = 0
    skipped = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(n):
            row = df.iloc[i]
            question = row[qcol]
            answer = row[acol]
            if pd.isna(question) or pd.isna(answer) or str(question).strip() == "" or str(answer).strip() == "":
                skipped += 1
                continue
            content = str(question).strip()
            if query_suffix:
                content = content + "\n" + query_suffix.strip()
            messages = [{"role": "user", "content": content}]
            solution = str(answer).strip()

            image_paths = []
            if icol and icol in df.columns:
                val = row[icol]
                # val 可能是 numpy 数组，不能直接 if pd.notna(val)，否则报 truth value ambiguous
                if val is None:
                    pass
                elif hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    if len(val) > 0:
                        val = val  # 下面按 list 处理
                    else:
                        val = None
                else:
                    try:
                        if not bool(pd.notna(val)):
                            val = None
                    except (ValueError, TypeError):
                        val = None
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        image_paths = list(val)
                    elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                        image_paths = list(val)
                    elif isinstance(val, str):
                        s = val.strip()
                        if not s:
                            pass
                        elif s.startswith("[") and "]" in s:
                            try:
                                import ast
                                image_paths = ast.literal_eval(s)
                                if not isinstance(image_paths, list):
                                    image_paths = [image_paths]
                            except Exception:
                                image_paths = [s]
                        else:
                            image_paths = [s]
                    else:
                        image_paths = [str(val)]
                resolved = []
                for p in image_paths:
                    p = str(p).strip()
                    if not p:
                        continue
                    if not os.path.isabs(p):
                        full = images_base / p
                        if not full.exists() and os.path.basename(p) != p:
                            full = images_base / os.path.basename(p)
                        p = str(full)
                    resolved.append(p)
                image_paths = resolved

            record = {"messages": messages, "solution": solution}
            if image_paths:
                record["images"] = image_paths
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"[Step 3] 已写入 {converted} 条到 {out_file}，跳过 {skipped} 条")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="ViRL39K 渐进式处理：下载 → 查看字段 → 转 GRPO 格式")
    parser.add_argument("--step", choices=["download", "inspect", "convert", "all"], default="all",
                        help="download=仅下载, inspect=仅查看字段, convert=仅转换, all=全部")
    parser.add_argument("--out_dir", type=str, default="./data/virl39k",
                        help="输出目录，将生成 train.jsonl、virl39k_schema.txt 等")
    parser.add_argument("--parquet", type=str, default=None,
                        help="已有 parquet 路径（convert 时必填或由 download 生成）")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="图片目录，默认 out_dir/images")
    parser.add_argument("--use_hf", action="store_true", default=True,
                        help="从 HuggingFace 下载（默认 True）")
    parser.add_argument("--no_use_hf", action="store_false", dest="use_hf", help="不从 HF 下载")
    # 转换时的列名映射（若与推断不一致可指定）
    parser.add_argument("--question_col", type=str, default=None)
    parser.add_argument("--answer_col", type=str, default=None)
    parser.add_argument("--image_col", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="最多转换条数，默认全部")
    parser.add_argument("--query_suffix", type=str, default="",
                        help="追加到每条 user content 的提示，如 <think> </think> 与 <answer> 说明")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    parquet_path = Path(args.parquet) if args.parquet else (out_dir / "39Krelease.parquet")
    images_dir = Path(args.images_dir) if args.images_dir else (out_dir / "images")

    if args.step in ("download", "all"):
        p, im = step_download(out_dir, use_hf=args.use_hf)
        if p is not None:
            parquet_path = p
        if im is not None:
            images_dir = im

    if args.step in ("inspect", "all"):
        step_inspect(parquet_path, images_dir, out_dir)

    if args.step in ("convert", "all"):
        if parquet_path.exists():
            # 一律从本地 parquet 转换（下载阶段已用 hf_hub_download 落盘，不再使用 load_dataset）
            step_convert(
                parquet_path=parquet_path,
                out_dir=out_dir,
                images_dir=images_dir,
                question_col=args.question_col,
                answer_col=args.answer_col,
                image_col=args.image_col,
                max_samples=args.max_samples,
                query_suffix=args.query_suffix,
            )
        else:
            print("[Step 3] 未找到 parquet，跳过 convert。请先执行 --step download 或指定 --parquet")


if __name__ == "__main__":
    main()
