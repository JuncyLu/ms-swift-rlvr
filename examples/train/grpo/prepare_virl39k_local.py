#!/usr/bin/env python3
"""Download ViRL39K from ModelScope (魔塔) and convert to ms-swift GRPO local format (train.jsonl + images/)."""
import argparse
import json
import os
import re
import shutil
from typing import Any, Optional

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download ViRL39K from ModelScope and prepare for ms-swift GRPO local training.')
    parser.add_argument('--output_dir', required=True, help='Output directory path.')
    parser.add_argument('--dataset_id', default='TIGER-Lab/ViRL39K',
                        help='ModelScope dataset id (default: TIGER-Lab/ViRL39K).')
    parser.add_argument('--subset_name', default='default',
                        help='Subset name on ModelScope (default: default).')
    parser.add_argument('--cache_dir', default=None, help='Optional cache dir for ModelScope.')
    parser.add_argument('--max_rows', type=int, default=0, help='0 for all, otherwise first N rows.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output_dir.')
    parser.add_argument('--use_hf', action='store_true',
                        help='Use HuggingFace instead of ModelScope when dataset not on 魔塔.')
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


def get_first_image(row: dict) -> Any:
    """ViRL39K 可能使用 'image' 或 'images' 列."""
    for key in ('image', 'images'):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        return val
    return None


def normalize_solution_for_reward(raw_answer: str) -> str:
    """Normalize ViRL39K answer (e.g. \\boxed{A}) to <answer>...</answer> for MultiModalAccuracyORM.
    So ground_truth from solution matches model output <answer>X</answer> (string comparison)."""
    raw = (raw_answer or '').strip()
    m = re.search(r'\\boxed\{([^}]*)\}', raw)
    inner = m.group(1).strip() if m else raw
    return f'<answer>{inner}</answer>' if inner else '<answer></answer>'


def load_dataset_from_modelscope(dataset_id: str, subset_name: str, cache_dir: Optional[str]) -> Any:
    """Load dataset from ModelScope (魔塔). Returns an iterable dataset."""
    from modelscope import MsDataset
    kwargs = {
        'subset_name': subset_name,
        'split': 'train',
        'version': 'master',
        'download_mode': 'reuse_dataset_if_exists',
    }
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    # MsDataset.load in newer versions may need trust_remote_code
    try:
        import modelscope
        if getattr(modelscope, '__version__', '0') >= '1.29.1':
            kwargs['trust_remote_code'] = True
    except Exception:
        pass
    ds = MsDataset.load(dataset_id, **kwargs)
    # ModelScope 返回的对象可能带有 _hf_ds（底层 HuggingFace Dataset）
    if hasattr(ds, '_hf_ds'):
        return ds._hf_ds
    if hasattr(ds, 'to_hf_dataset'):
        return ds.to_hf_dataset()
    return ds


def load_dataset_from_huggingface(dataset_id: str, cache_dir: Optional[str]) -> Any:
    """Load dataset from HuggingFace (fallback when not on ModelScope)."""
    from datasets import load_dataset
    kwargs = {'split': 'train'}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    return load_dataset(dataset_id, **kwargs)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, 'images')
    train_path = os.path.join(output_dir, 'train.jsonl')

    ensure_empty_dir(output_dir, args.overwrite)
    os.makedirs(images_dir, exist_ok=True)

    if args.use_hf:
        print('Loading dataset from HuggingFace...')
        ds = load_dataset_from_huggingface(args.dataset_id, args.cache_dir)
    else:
        print('Loading dataset from ModelScope (魔塔)...')
        ds = load_dataset_from_modelscope(args.dataset_id, args.subset_name, args.cache_dir)

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
