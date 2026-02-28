#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import shutil
from typing import Iterable, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare TIGER-Lab/ViRL39K for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--cache_dir', default=None, help='Optional HF cache dir.')
    parser.add_argument('--split', default='train', help='Dataset split to use.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    return parser.parse_args()


def ensure_empty_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f'Output dir exists: {path}. Use --overwrite to replace.')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sanitize_id(raw: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(raw))


def _candidate_base_dirs(ds) -> List[str]:
    base_dirs = set()
    for item in getattr(ds, 'cache_files', []) or []:
        filename = item.get('filename')
        if filename:
            base_dirs.add(os.path.dirname(filename))
    return [d for d in base_dirs if d]


def _open_image_from_any(img, base_dirs: Iterable[str]) -> Optional[Image.Image]:
    if img is None:
        return None
    if hasattr(img, 'save'):
        return img
    if isinstance(img, dict):
        if 'bytes' in img and img['bytes'] is not None:
            return Image.open(io.BytesIO(img['bytes']))
        if 'path' in img and img['path']:
            img = img['path']
    if isinstance(img, str):
        if os.path.exists(img):
            return Image.open(img)
        for base in base_dirs:
            candidate = os.path.join(base, img)
            if os.path.exists(candidate):
                return Image.open(candidate)
    return None


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
    os.makedirs(images_dir, exist_ok=True)

    ds = load_dataset('TIGER-Lab/ViRL39K', split=args.split, cache_dir=args.cache_dir)
    total = len(ds)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else total
    base_dirs = _candidate_base_dirs(ds)

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
                pil = _open_image_from_any(img, base_dirs)
                if pil is None:
                    continue
                if pil.mode != 'RGB':
                    pil = pil.convert('RGB')
                img_filename = f'{qid_safe}-{j}.jpg'
                img_path = os.path.join(images_dir, img_filename)
                pil.save(img_path, quality=95)
                image_paths.append(img_path)

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
