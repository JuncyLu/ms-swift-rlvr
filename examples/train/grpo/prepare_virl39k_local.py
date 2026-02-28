#!/usr/bin/env python3
"""Download TIGER-Lab/ViRL39K and convert to ms-swift GRPO local format (train.jsonl + images/)."""
import argparse
import json
import os
import re
import shutil
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ViRL39K (TIGER-Lab/ViRL39K) for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--cache_dir', default=None, help='Optional HF cache dir.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    parser.add_argument('--num_proc', type=int, default=1, help='Number of processes (unused).')
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


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, 'images')
    train_path = os.path.join(output_dir, 'train.jsonl')

    ensure_empty_dir(output_dir, args.overwrite)
    os.makedirs(images_dir, exist_ok=True)

    ds = load_dataset('TIGER-Lab/ViRL39K', split='train', cache_dir=args.cache_dir)
    total = len(ds)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else total

    with open(train_path, 'w', encoding='utf-8') as f:
        for i, row in enumerate(tqdm(ds, total=min(total, max_rows), desc='Processing')):
            if i >= max_rows:
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
