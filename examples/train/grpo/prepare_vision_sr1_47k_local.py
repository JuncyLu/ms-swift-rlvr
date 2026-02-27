#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Vision-SR1-47K for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--cache_dir', default=None, help='Optional HF cache dir.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    parser.add_argument('--num_proc', type=int, default=1, help='Number of processes (not used yet).')
    return parser.parse_args()


def ensure_empty_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f'Output dir exists: {path}. Use --overwrite to replace.')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_image(image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, 'images')
    train_path = os.path.join(output_dir, 'train.jsonl')

    ensure_empty_dir(output_dir, args.overwrite)
    os.makedirs(images_dir, exist_ok=True)

    ds = load_dataset('LMMs-Lab-Turtle/Vision-SR1-47K', split='train', cache_dir=args.cache_dir)
    total = len(ds)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else total

    with open(train_path, 'w', encoding='utf-8') as f:
        for i, row in enumerate(tqdm(ds, total=min(total, max_rows), desc='Processing')):
            if i >= max_rows:
                break
            problem_id = row.get('problem_id')
            if problem_id is None or str(problem_id).strip() == '':
                problem_id = str(i)
            img = row['images']
            img_filename = f'{problem_id}.png'
            img_path = os.path.join(images_dir, img_filename)
            save_image(img, img_path)

            record = {
                'messages': [{'role': 'user', 'content': row['problem']}],
                'images': img_path,
                'solution': row['answer'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'train.jsonl: {train_path}')
    print('Example:')
    print(f'  --dataset {train_path}')


if __name__ == '__main__':
    main()
