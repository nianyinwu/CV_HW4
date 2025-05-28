""" Split training data to training and validation """

import os
import random
import argparse
import shutil

def get_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--src_root',
        '-s',
        type=str,
        default='../datas/original_train',
        help='the path of source root'
    )
    parser.add_argument(
        '--dst_root',
        '-d',
        type=str,
        default='../datas',
        help='the path of destination root'
    )

    return parser.parse_args()


def make_dirs(base_path):
    """
    Create folder
    """

    os.makedirs(os.path.join(base_path, "clean"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "degraded"), exist_ok=True)

def split_and_move(src_root, dst_root, valid_ratio=0.2, seed=42):
    """
    Split data and move to correct folder
    """

    random.seed(seed)
    total = 1600
    rain_ids = list(range(1, total + 1))
    snow_ids = list(range(1, total + 1))
    random.shuffle(rain_ids)
    random.shuffle(snow_ids)

    val_num = int(total * valid_ratio)

    split_map = {
        'train': {
            'rain': rain_ids[val_num:],
            'snow': snow_ids[val_num:]
        },
        'valid': {
            'rain': rain_ids[:val_num],
            'snow': snow_ids[:val_num]
        }
    }

    for split in ['train', 'valid']:
        for category in ['rain', 'snow']:
            for idx in split_map[split][category]:
                # file names
                degraded_name = f"{category}-{idx}.png"
                clean_name = f"{category}_clean-{idx}.png"

                # source path
                degraded_src = os.path.join(src_root, "degraded", degraded_name)
                clean_src = os.path.join(src_root, "clean", clean_name)

                # destination path
                degraded_dst = os.path.join(dst_root, split, 'degraded', degraded_name)
                clean_dst = os.path.join(dst_root, split, 'clean', clean_name)

                # copy files
                shutil.copy(degraded_src, degraded_dst)
                shutil.copy(clean_src, clean_dst)

if __name__ == "__main__":
    opt = get_args()

    # create folders
    root = os.path.dirname(opt.src_root)
    make_dirs(os.path.join(root, "train"))
    make_dirs(os.path.join(root, "valid"))

    # move data
    split_and_move(opt.src_root, opt.dst_root)
    print("Finished !!!")
