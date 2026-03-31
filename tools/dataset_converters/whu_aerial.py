"""
python datasets/whu_aerial.py data/whu_aerial/val -o data/whu_aerial/val
"""
import argparse
import os
import os.path as osp
import shutil
import cv2

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert WHU_Aerial dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='WHU_Aerial folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'WHU_Aerial')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'label_cvt'))

    label_path = osp.join(dataset_path, 'label')
    for img_name in os.listdir(label_path):
        img = cv2.imread(osp.join(label_path, img_name), cv2.IMREAD_GRAYSCALE)
        img[img < 128] = 0
        img[img >= 128] = 1
        cv2.imwrite(osp.join(out_dir, 'label_cvt', img_name), img)
        
    print('Done!')


if __name__ == '__main__':
    main()