#!/usr/bin/env python

import os
import sys
import json
import torch
import argparse
import torchvision

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data.loader_track import build_video_info
from core.libs.utils_lmdb import LMDBEngine


def export_feature_frames(dataset_path, output_dir, split='train'):
    dataset_json_path = os.path.join(dataset_path, 'dataset.json')
    lmdb_path = os.path.join(dataset_path, 'img_lmdb')
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f'dataset.json not found: {dataset_json_path}')
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f'img_lmdb not found: {lmdb_path}')

    with open(dataset_json_path, 'r') as file_obj:
        dataset_split = json.load(file_obj)
    if split not in dataset_split:
        raise KeyError(f'Split {split} not found in dataset.json')

    video_info, _ = build_video_info(dataset_split[split])
    feature_keys = [video_info[video_id][0] for video_id in sorted(video_info.keys())]

    os.makedirs(output_dir, exist_ok=True)
    lmdb_engine = LMDBEngine(lmdb_path, write=False)
    try:
        for frame_key in feature_keys:
            image = lmdb_engine[frame_key]
            output_path = os.path.join(output_dir, f'{frame_key}.png')
            torchvision.io.write_png(image.to(torch.uint8), output_path)
    finally:
        lmdb_engine.close()

    manifest_path = os.path.join(output_dir, 'manifest.txt')
    with open(manifest_path, 'w') as file_obj:
        for frame_key in feature_keys:
            file_obj.write(f'{frame_key}\n')

    print(f'Exported {len(feature_keys)} feature frames to {output_dir}.')
    print(f'Manifest saved to {manifest_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--split', default='train', type=str)
    args = parser.parse_args()
    export_feature_frames(args.dataset_path, args.output_dir, split=args.split)