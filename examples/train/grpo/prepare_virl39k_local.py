#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare TIGER-Lab/ViRL39K for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--cache_dir', default=None, help='Optional HF cache dir.')
    parser.add_argument('--repo_id', default='TIGER-Lab/ViRL39K', help='HF dataset repo id.')
    parser.add_argument('--split', default='train', help='Dataset split to use.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    parser.add_argument(
        '--image_mode',
        default='as_is',
        choices=['as_is', 'symlink', 'copy'],
        help='How to handle images: as_is (use paths in HF cache), symlink or copy into output_dir/images.')
    return parser.parse_args()


def ensure_empty_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f'Output dir exists: {path}. Use --overwrite to replace.')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sanitize_id(raw: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(raw))


def _find_parquet(repo_dir: str) -> str:
    parquet_files: List[Tuple[int, str]] = []
    for root, _, files in os.walk(repo_dir):
        for name in files:
            if name.endswith('.parquet'):
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                parquet_files.append((size, path))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found under: {repo_dir}')
    parquet_files.sort(reverse=True)
    return parquet_files[0][1]


def _verify_parquet_magic(path: str) -> None:
    # Parquet files start and end with magic bytes "PAR1"
    with open(path, 'rb') as f:
        head = f.read(4)
        if head != b'PAR1':
            raise RuntimeError(
                f'Parquet magic bytes not found in header: {path}. '
                'This usually means the dataset was downloaded as a pointer file. '
                'Please upgrade huggingface_hub (>=0.32) or install hf-xet, then retry.')
        try:
            f.seek(-4, os.SEEK_END)
        except OSError:
            raise RuntimeError(
                f'Parquet file too small or invalid: {path}. '
                'Please upgrade huggingface_hub (>=0.32) or install hf-xet, then retry.')
        tail = f.read(4)
        if tail != b'PAR1':
            raise RuntimeError(
                f'Parquet magic bytes not found in footer: {path}. '
                'Please upgrade huggingface_hub (>=0.32) or install hf-xet, then retry.')


def normalize_question(question: str) -> str:
    if not question:
        return question
    # ViRL39K questions often start with "<image>"
    q = question.strip()
    if q.lower().startswith('<image>'):
        q = q[len('<image>'):].strip()
    return q


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, 'images')
    train_path = os.path.join(output_dir, 'train.jsonl')

    ensure_empty_dir(output_dir, args.overwrite)

    repo_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type='dataset',
        cache_dir=args.cache_dir,
    )
    parquet_path = _find_parquet(repo_dir)
    _verify_parquet_magic(parquet_path)
    ds = load_dataset('parquet', data_files=parquet_path, split=args.split)

    if args.image_mode in ('symlink', 'copy'):
        os.makedirs(images_dir, exist_ok=True)
    total = len(ds)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else total
    with open(train_path, 'w', encoding='utf-8') as f:
        for i, row in enumerate(tqdm(ds, total=min(total, max_rows), desc='Processing')):
            if i >= max_rows:
                break
            qid = row.get('qid') or i
            qid_safe = sanitize_id(qid)
            question = row.get('question') or row.get('prompt') or row.get('query') or row.get('problem')
            answer = row.get('answer') or row.get('solution') or row.get('response')
            if question is None or answer is None:
                continue
            question = normalize_question(question)

            images = row.get('image') or row.get('images')
            if images is None:
                continue
            if not isinstance(images, list):
                images = [images]

            image_paths: List[str] = []
            for j, img in enumerate(images):
                if isinstance(img, dict) and 'path' in img:
                    img = img['path']
                if not isinstance(img, str):
                    continue
                src_path = img if os.path.isabs(img) else os.path.join(repo_dir, img)
                if not os.path.exists(src_path):
                    continue
                if args.image_mode == 'as_is':
                    image_paths.append(src_path)
                else:
                    img_filename = f'{qid_safe}-{j}{os.path.splitext(src_path)[1] or ".jpg"}'
                    dst_path = os.path.join(images_dir, img_filename)
                    if args.image_mode == 'symlink':
                        if not os.path.exists(dst_path):
                            os.symlink(src_path, dst_path)
                    else:
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                    image_paths.append(dst_path)

            if not image_paths:
                continue

            record = {
                'messages': [{'role': 'user', 'content': question}],
                'images': image_paths if len(image_paths) > 1 else image_paths[0],
                'solution': answer,
                'qid': qid,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'train.jsonl: {train_path}')
    print('Example:')
    print(f'  --dataset {train_path}')


if __name__ == '__main__':
    main()
