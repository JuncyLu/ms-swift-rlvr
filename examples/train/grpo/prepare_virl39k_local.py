#!/usr/bin/env python3
"""Download TIGER-Lab/ViRL39K and convert to ms-swift GRPO local format (train.jsonl + images/)."""
import argparse
import json
import os
import re
import shutil
import zipfile
from typing import Any, Iterator, Tuple

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ViRL39K (TIGER-Lab/ViRL39K) for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--cache_dir', default=None, help='Optional HF cache dir.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    parser.add_argument('--num_proc', type=int, default=1, help='Number of processes (unused).')
    parser.add_argument(
        '--parquet_path',
        default=None,
        help='Local path to 39Krelease.parquet. Use with --images_zip_path to avoid HF parquet buffer error.',
    )
    parser.add_argument(
        '--images_zip_path',
        default=None,
        help='Local path to images.zip. Use with --parquet_path for local loading.',
    )
    return parser.parse_args()


def ensure_empty_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f'Output dir exists: {path}. Use --overwrite to replace.')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(s: str, max_len: int = 200) -> str:
    s = re.sub(r'[^\w\-.]', '_', s)[:max_len]
    return s or 'unnamed'


def save_image(image: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(image, 'save'):
        image.save(path)
        return
    if isinstance(image, str):
        from PIL import Image
        with open(image, 'rb') as fp:
            img = Image.open(fp).convert('RGB')
        img.save(path)
        return
    raise TypeError(f'Cannot save image type: {type(image)}')


def get_first_image(row: dict, key: str = 'image') -> Any:
    val = row.get(key)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val


def normalize_solution_for_reward(raw_answer: str) -> str:
    """Normalize ViRL39K answer (e.g. \\boxed{A}) to <answer>...</answer> for MultiModalAccuracyORM.
    So ground_truth from solution matches model output <answer>X</answer> (string comparison)."""
    raw = (raw_answer or '').strip()
    m = re.search(r'\\boxed\{([^}]*)\}', raw)
    inner = m.group(1).strip() if m else raw
    return f'<answer>{inner}</answer>' if inner else '<answer></answer>'


def _load_local_virl39k(
    parquet_path: str, images_zip_path: str, extract_dir: str
) -> Tuple[int, Iterator[dict]]:
    """Load ViRL39K from local 39Krelease.parquet + images.zip. Returns (num_rows, iterator of row dicts)."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    col_names = table.column_names
    n = table.num_rows

    with zipfile.ZipFile(images_zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    def _iter_rows():
        for i in range(n):
            row = {}
            for j, name in enumerate(col_names):
                col = table.column(j)
                if name == 'image':
                    val = col[i]
                    if val is None:
                        row[name] = []
                    else:
                        if hasattr(val, 'as_py'):
                            val = val.as_py()
                        if isinstance(val, (list, tuple)) and val:
                            first = val[0]
                            path_in_zip = None
                            if isinstance(first, str):
                                path_in_zip = first
                            elif isinstance(first, dict):
                                path_in_zip = first.get('path')
                                if path_in_zip is None and first.get('bytes'):
                                    row[name] = []  # in-buffer image not supported in local mode
                                    path_in_zip = False
                            if path_in_zip:
                                full = os.path.join(extract_dir, path_in_zip)
                                row[name] = [full] if os.path.isfile(full) else []
                            elif name not in row:
                                row[name] = list(val) if val else []
                        else:
                            row[name] = list(val) if val else []
                else:
                    v = col[i]
                    if hasattr(v, 'as_py'):
                        v = v.as_py()
                    row[name] = v
            yield row

    return n, _iter_rows()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, 'images')
    train_path = os.path.join(output_dir, 'train.jsonl')

    ensure_empty_dir(output_dir, args.overwrite)
    os.makedirs(images_dir, exist_ok=True)

    use_local = args.parquet_path and args.images_zip_path
    if use_local:
        if not os.path.isfile(args.parquet_path):
            raise FileNotFoundError(f'Parquet not found: {args.parquet_path}')
        if not os.path.isfile(args.images_zip_path):
            raise FileNotFoundError(f'Images zip not found: {args.images_zip_path}')
        extract_dir = os.path.join(output_dir, '_virl39k_images')
        os.makedirs(extract_dir, exist_ok=True)
        total, row_iter = _load_local_virl39k(
            args.parquet_path, args.images_zip_path, extract_dir
        )
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset('TIGER-Lab/ViRL39K', split='train', cache_dir=args.cache_dir)
            total = len(ds)
            row_iter = iter(ds)
        except Exception as e:
            err_msg = str(e) + (str(e.__cause__) if e.__cause__ else '')
            if 'Parquet magic bytes' in err_msg or 'ArrowInvalid' in err_msg:
                print('load_dataset failed (known parquet buffer issue). Use local files:')
                print('  1. Download from https://huggingface.co/datasets/TIGER-Lab/ViRL39K: 39Krelease.parquet, images.zip')
                print('  2. Run with: --parquet_path /path/to/39Krelease.parquet --images_zip_path /path/to/images.zip')
            raise

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else (total or 0)
    if total is not None and max_rows:
        max_rows = min(max_rows, total)
    tqdm_total = min(max_rows, total) if (max_rows and total) else (total or None)

    with open(train_path, 'w', encoding='utf-8') as f:
        for i, row in enumerate(tqdm(row_iter, total=tqdm_total, desc='Processing')):
            if max_rows and i >= max_rows:
                break
            img = get_first_image(row)
            if img is None:
                continue
            qid = row.get('qid') or str(i)
            img_filename = _sanitize_filename(str(qid)) + '.jpg'
            img_path = os.path.join(images_dir, img_filename)
            try:
                save_image(img, img_path)
            except Exception:
                continue
            question = row.get('question') or ''
            answer = row.get('answer') or ''
            record = {
                'messages': [{'role': 'user', 'content': question}],
                'images': img_path,
                'solution': normalize_solution_for_reward(answer),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'train.jsonl: {train_path}')
    print('Example:')
    print(f'  --dataset {train_path}')


if __name__ == '__main__':
    main()
