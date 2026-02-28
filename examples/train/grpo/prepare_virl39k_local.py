#!/usr/bin/env python3
"""Download ViRL39K from ModelScope (魔塔) and convert to ms-swift GRPO local format (train.jsonl + images/)."""
import argparse
import json
import os
import re
import shutil
import zipfile
from typing import Any, Iterator, Optional

import requests
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


def _download_ms_dataset_file(dataset_id: str, file_path: str, local_path: str) -> None:
    """从魔塔下载数据集中的单个文件（二进制安全）。"""
    from modelscope.hub.api import ModelScopeConfig
    url = (
        f'https://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo'
        f'?Source=SDK&Revision=master&FilePath={file_path}'
    )
    cookies = ModelScopeConfig.get_cookies()
    resp = requests.get(url, cookies=cookies, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    total = int(resp.headers.get('content-length', 0)) or None
    with open(local_path, 'wb') as f:
        with tqdm(total=total, desc=file_path, unit='B', unit_scale=True, leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def load_dataset_from_modelscope_manual(
    dataset_id: str,
    cache_dir: Optional[str],
    max_rows: int,
) -> Iterator[dict]:
    """
    直接从魔塔下载 39Krelease.parquet 和 images.zip，本地解析，避免 MsDataset 内部
    parquet 解析错误（Parquet magic bytes not found）。
    """
    import pandas as pd

    cache_base = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', 'datasets', 'virl39k_manual')
    os.makedirs(cache_base, exist_ok=True)
    parquet_path = os.path.join(cache_base, '39Krelease.parquet')
    zip_path = os.path.join(cache_base, 'images.zip')
    extracted_dir = os.path.join(cache_base, 'images_extracted')

    for fpath, local in [('39Krelease.parquet', parquet_path), ('images.zip', zip_path)]:
        if not os.path.exists(local) or os.path.getsize(local) == 0:
            _download_ms_dataset_file(dataset_id, fpath, local)

    if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
        print('Extracting images.zip...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extracted_dir)

    print('Loading parquet with pandas...')
    df = pd.read_parquet(parquet_path)
    total = len(df)
    n = min(total, max_rows) if max_rows > 0 else total

    for i in range(n):
        row = df.iloc[i]
        qid = str(row.get('qid', i))
        question = str(row.get('question', ''))
        answer = str(row.get('answer', ''))
        img = row.get('image') or row.get('images')
        if img is None:
            continue
        # ViRL39K parquet: image 为路径列表，如 ["images/Processed-xxx-0.jpg"] 或 HF Image 结构 {"path": "..."}
        if isinstance(img, (list, tuple)) and len(img) > 0:
            first = img[0]
            if isinstance(first, dict):
                path_in_zip = first.get('path')
                if path_in_zip is None:
                    continue  # 仅有 bytes 的 HF Image 暂不处理
                path_in_zip = str(path_in_zip)
            else:
                path_in_zip = str(first)
        else:
            path_in_zip = str(img)
        if not path_in_zip:
            continue
        src = os.path.join(extracted_dir, path_in_zip)
        if not os.path.isfile(src):
            continue
        yield {'qid': qid, 'question': question, 'answer': answer, 'image_path': src}


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
    try:
        import modelscope
        if getattr(modelscope, '__version__', '0') >= '1.29.1':
            kwargs['trust_remote_code'] = True
    except Exception:
        pass
    ds = MsDataset.load(dataset_id, **kwargs)
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

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else 0

    if args.use_hf:
        print('Loading dataset from HuggingFace...')
        ds = load_dataset_from_huggingface(args.dataset_id, args.cache_dir)
        total = len(ds)
        n = min(total, max_rows) if max_rows else total
        with open(train_path, 'w', encoding='utf-8') as f:
            for i, row in enumerate(tqdm(ds, total=n, desc='Processing')):
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
    else:
        print('Loading dataset from ModelScope (魔塔) - manual parquet+zip...')
        it = load_dataset_from_modelscope_manual(args.dataset_id, args.cache_dir, max_rows)
        count = 0
        with open(train_path, 'w', encoding='utf-8') as f:
            for row in tqdm(it, desc='Processing'):
                qid = row['qid']
                img_filename = _sanitize_filename(qid) + '.jpg'
                dst_path = os.path.join(images_dir, img_filename)
                try:
                    shutil.copy2(row['image_path'], dst_path)
                except Exception:
                    continue
                record = {
                    'messages': [{'role': 'user', 'content': row['question']}],
                    'images': dst_path,
                    'solution': normalize_solution_for_reward(row['answer']),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        total = count

    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'train.jsonl: {train_path}')
    print('Example:')
    print(f'  --dataset {train_path}')


if __name__ == '__main__':
    main()
