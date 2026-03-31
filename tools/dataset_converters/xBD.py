"""
python datasets/cvt_xBD.py data/rs_seg/Building/xBD/test -o data/rs_seg/Building/xBD/test

"""
import argparse
import os
import os.path as osp
import shutil

import cv2
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert xBD dataset and separate pre-disaster files.')
    parser.add_argument('dataset_path', help='xBD folder path (e.g., data/xBD/test)')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args

def copy_pre_disaster_files(src_dir, dest_dir, keyword='pre_disaster'):
    if not osp.exists(src_dir):
        print(f"Warning: Source directory not found, skipping: {src_dir}")
        return 0
        
    mkdir_or_exist(dest_dir)
    count = 0
    print(f"Copying files from '{src_dir}' to '{dest_dir}'...")
    for filename in os.listdir(src_dir):
        if keyword in filename:
            src_file_path = osp.join(src_dir, filename)
            dest_file_path = osp.join(dest_dir, filename)
            shutil.copy(src_file_path, dest_file_path)
            count += 1
    print(f"Copied {count} files.")
    return count


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = dataset_path
    else:
        out_dir = args.out_dir

    print('--- Step 1: Converting target files ---')
    print('Making directories...')
    
    targets_cvt_dir = osp.join(out_dir, 'targets_cvt')
    mkdir_or_exist(targets_cvt_dir)

    targets_src_dir = osp.join(dataset_path, 'targets')
    for img_name in os.listdir(targets_src_dir):
        img_path = osp.join(targets_src_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
            
        img[img >= 1] = 1
        
        new_img_name = img_name.replace('_target', '')
        cv2.imwrite(osp.join(targets_cvt_dir, new_img_name), img)

    print('Target conversion finished!')
    print('-' * 40)

    print('--- Step 2: Separating pre-disaster files by copying ---')
    images_src_dir = osp.join(dataset_path, 'images')
    images_pre_dest_dir = osp.join(out_dir, 'images_pre')
    
    targets_pre_dest_dir = osp.join(out_dir, 'targets_cvt_pre')

    copy_pre_disaster_files(images_src_dir, images_pre_dest_dir)
    copy_pre_disaster_files(targets_cvt_dir, targets_pre_dest_dir)

    print('-' * 40)
    print('All Done!')


if __name__ == '__main__':
    main()