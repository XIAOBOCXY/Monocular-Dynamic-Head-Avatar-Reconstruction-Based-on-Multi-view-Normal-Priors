#!/usr/bin/env python

import argparse
import os
import shutil
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.libs.utils_lmdb import LMDBEngine
from core.libs.GAGAvatar_track.engines import CoreEngine
from core.libs.GAGAvatar_track.engines.human_matting import StyleMatteEngine


def matte_lmdb(src_dataset, dst_dataset, device='cuda', background=0.0, skip_existing=False):
    src_lmdb_path = os.path.join(src_dataset, 'img_lmdb')
    dst_lmdb_path = os.path.join(dst_dataset, 'img_lmdb')
    if not os.path.isdir(src_lmdb_path):
        raise FileNotFoundError(f'img_lmdb not found: {src_lmdb_path}')

    os.makedirs(dst_dataset, exist_ok=True)
    src_lmdb = LMDBEngine(src_lmdb_path, write=False)
    dst_lmdb = LMDBEngine(dst_lmdb_path, write=True)
    matting_engine = StyleMatteEngine(device=device)

    processed = 0
    try:
        all_keys = sorted(src_lmdb.keys())
        for key in tqdm(all_keys, desc='Matting img_lmdb'):
            if skip_existing and dst_lmdb.exists(key):
                continue
            image = src_lmdb[key].float().to(device) / 255.0
            image = matting_engine(image, return_type='matting', background_rgb=background)
            image = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
            if dst_lmdb.exists(key):
                dst_lmdb.delete(key)
            dst_lmdb.dump(key, image, type='image')
            processed += 1
    finally:
        src_lmdb.close()
        dst_lmdb.close()
    return processed


def copy_split_file(src_dataset, dst_dataset):
    src_json = os.path.join(src_dataset, 'dataset.json')
    if os.path.exists(src_json):
        shutil.copy2(src_json, os.path.join(dst_dataset, 'dataset.json'))


def copy_tracking_files(src_dataset, dst_dataset):
    for file_name in ['base.pkl', 'optim.pkl']:
        src_path = os.path.join(src_dataset, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_dataset, file_name))


def retrack_dataset(dataset_path, device='cuda', focal=12.0):
    lmdb_path = os.path.join(dataset_path, 'img_lmdb')
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f'img_lmdb not found: {lmdb_path}')
    for file_name in ['base.pkl', 'optim.pkl']:
        file_path = os.path.join(dataset_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
    engine = CoreEngine(focal_length=focal, device=device)
    lmdb_engine = LMDBEngine(lmdb_path, write=False)
    try:
        base_results = engine.track_base(lmdb_engine, dataset_path)
        optim_results = engine.track_optim(base_results, dataset_path, lmdb_engine, share_id=False)
    finally:
        lmdb_engine.close()
    return len(base_results), len(optim_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dataset', required=True, type=str)
    parser.add_argument('--dst_dataset', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--background', default=0.0, type=float)
    parser.add_argument('--focal', default=12.0, type=float)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--copy_tracking', action='store_true')
    parser.add_argument('--rerun_track', action='store_true')
    args = parser.parse_args()

    src_dataset = os.path.abspath(args.src_dataset)
    dst_dataset = os.path.abspath(args.dst_dataset)
    if src_dataset == dst_dataset:
        raise ValueError('src_dataset and dst_dataset must be different paths.')

    processed = matte_lmdb(
        src_dataset=src_dataset,
        dst_dataset=dst_dataset,
        device=args.device,
        background=args.background,
        skip_existing=args.skip_existing,
    )
    copy_split_file(src_dataset, dst_dataset)

    if args.rerun_track:
        base_count, optim_count = retrack_dataset(dst_dataset, device=args.device, focal=args.focal)
        print(
            f'Done. matted={processed} base={base_count} optim={optim_count} dst_dataset={dst_dataset}'
        )
        return

    if args.copy_tracking:
        copy_tracking_files(src_dataset, dst_dataset)

    print(f'Done. matted={processed} dst_dataset={dst_dataset}')


if __name__ == '__main__':
    main()