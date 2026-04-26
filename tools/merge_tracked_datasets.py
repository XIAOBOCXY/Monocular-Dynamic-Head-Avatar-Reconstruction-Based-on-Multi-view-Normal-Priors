#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Ensure project root is on sys.path so `core` can be imported
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.libs.utils_lmdb import LMDBEngine

def _load_pickle(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_dataset_json(
    path: str,
    fallback_keys: List[str],
    train_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        # Normalize missing splits
        if 'train' not in data:
            data['train'] = []
        if 'val' not in data:
            data['val'] = []
        if 'test' not in data:
            data['test'] = []
        return data
    # Fallback: split keys when no dataset.json is present
    keys = list(fallback_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    split_idx = int(len(keys) * train_ratio)
    return {'train': keys[:split_idx], 'val': keys[split_idx:], 'test': []}

def _normalize_prefix(prefix: str) -> str:
    if prefix is None:
        return ''
    prefix = prefix.strip()
    if not prefix:
        return ''
    if not prefix.endswith('_'):
        prefix = prefix + '_'
    return prefix

def _parse_csv(value: str) -> List[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]

def _collect_dataset_dirs(parent_dir: str) -> List[str]:
    if not parent_dir or not os.path.isdir(parent_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(parent_dir)):
        child = os.path.join(parent_dir, name)
        if not os.path.isdir(child):
            continue
        if os.path.exists(os.path.join(child, 'optim.pkl')) and os.path.exists(os.path.join(child, 'img_lmdb')):
            dirs.append(child)
    return dirs

def _merge_one_dataset(
    src_dir: str,
    prefix: str,
    dst_lmdb: Optional[LMDBEngine],
    dst_optim: Dict,
    dst_splits: Dict[str, List[str]],
    on_conflict: str,
    dry_run: bool,
    train_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    optim_path = os.path.join(src_dir, 'optim.pkl')
    lmdb_path = os.path.join(src_dir, 'img_lmdb')
    dataset_path = os.path.join(src_dir, 'dataset.json')

    if not os.path.exists(optim_path):
        raise FileNotFoundError(f'optim.pkl not found in {src_dir}')
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f'img_lmdb not found in {src_dir}')

    src_optim = _load_pickle(optim_path)
    src_splits = _load_dataset_json(
        dataset_path,
        list(src_optim.keys()),
        train_ratio=train_ratio,
        seed=seed,
    )

    src_lmdb = LMDBEngine(lmdb_path, write=False)
    copied = 0
    skipped = 0
    copied_keys = set()

    try:
        for key in tqdm(list(src_optim.keys()), desc=f'Copy {os.path.basename(src_dir)}'):
            new_key = f'{prefix}{key}'
            if new_key in dst_optim:
                if on_conflict == 'skip':
                    skipped += 1
                    continue
                raise ValueError(f'Key conflict: {new_key} from {src_dir}')

            raw_payload = src_lmdb.raw_load(key)
            if raw_payload is None:
                skipped += 1
                continue

            if not dry_run:
                if dst_lmdb is None:
                    raise RuntimeError('dst_lmdb is required when not in dry_run mode.')
                dst_lmdb.raw_dump(new_key, raw_payload)
                dst_optim[new_key] = src_optim[key]
            copied_keys.add(new_key)
            copied += 1
    finally:
        src_lmdb.close()

    for split_name in ['train', 'val', 'test']:
        split_keys = [
            f'{prefix}{key}' for key in src_splits.get(split_name, []) if f'{prefix}{key}' in copied_keys
        ]
        if not dry_run:
            dst_splits[split_name].extend(split_keys)

    return copied, skipped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset dirs')
    parser.add_argument('--dataset', action='append', default=[], help='Dataset dir (repeatable)')
    parser.add_argument('--datasets_dir', action='append', default=[], help='Parent dir that contains many dataset subdirs')
    parser.add_argument('--prefixes', type=str, default=None, help='Comma-separated prefixes')
    parser.add_argument('--prefix', action='append', default=[], help='Prefix per dataset (repeatable)')
    parser.add_argument('--out_dir', required=True, type=str, help='Output dataset dir')
    parser.add_argument('--on_conflict', choices=['error', 'skip'], default='error')
    parser.add_argument('--mode', choices=['new', 'append'], default='new')
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    datasets = []
    datasets.extend(_parse_csv(args.datasets))
    datasets.extend(args.dataset)
    for parent in args.datasets_dir:
        datasets.extend(_collect_dataset_dirs(parent))
    datasets = [os.path.abspath(d) for d in datasets if d]

    prefixes = []
    prefixes.extend(_parse_csv(args.prefixes))
    prefixes.extend(args.prefix)
    prefixes = [_normalize_prefix(p) for p in prefixes]

    if not datasets:
        raise ValueError('No datasets provided. Use --datasets or --dataset.')

    if prefixes and len(prefixes) != len(datasets):
        raise ValueError('prefixes count must match datasets count if provided.')

    if not prefixes:
        prefixes = ['' for _ in datasets]

    os.makedirs(args.out_dir, exist_ok=True)

    dst_optim = {}
    dst_splits = {'train': [], 'val': [], 'test': []}
    dst_lmdb = None

    if args.mode == 'append':
        optim_path = os.path.join(args.out_dir, 'optim.pkl')
        dataset_path = os.path.join(args.out_dir, 'dataset.json')
        if os.path.exists(optim_path):
            dst_optim = _load_pickle(optim_path)
        if os.path.exists(dataset_path):
            dst_splits = _load_dataset_json(
                dataset_path,
                list(dst_optim.keys()),
                train_ratio=args.train_ratio,
                seed=args.seed,
            )

    if not args.dry_run:
        dst_lmdb = LMDBEngine(os.path.join(args.out_dir, 'img_lmdb'), write=True)

    total_copied = 0
    total_skipped = 0
    try:
        for src_dir, prefix in zip(datasets, prefixes):
            copied, skipped = _merge_one_dataset(
                src_dir=src_dir,
                prefix=prefix,
                dst_lmdb=dst_lmdb,
                dst_optim=dst_optim,
                dst_splits=dst_splits,
                on_conflict=args.on_conflict,
                dry_run=args.dry_run,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
            total_copied += copied
            total_skipped += skipped

        if not args.dry_run:
            with open(os.path.join(args.out_dir, 'optim.pkl'), 'wb') as f:
                pickle.dump(dst_optim, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(args.out_dir, 'dataset.json'), 'w') as f:
                json.dump(dst_splits, f, indent=2)
    finally:
        if dst_lmdb is not None:
            dst_lmdb.close()
    print(f'Done. copied={total_copied} skipped={total_skipped} dry_run={args.dry_run}')

if __name__ == '__main__':
    main()
