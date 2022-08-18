#! /bin/python3

import sys
import os
import shutil


def main(input_dir, output_dir, name, range_start, range_end):
    if not (os.path.exists(input_dir) and os.path.exists(output_dir)):
        print(f'{input_dir} or {output_dir} does not exist!')
        return

    print(f'Moving files from {input_dir} to {output_dir}.')
    print(f'In range of {range_start} to {range_end}.')

    for i in range(range_start, range_end + 1):
        input_file = os.path.join(input_dir, f'{name}{i}.npy')
        output_file = os.path.join(output_dir, f'{name}{i}.npy')
        if os.path.exists(input_file):
            shutil.move(input_file, output_file)
        else:
            print(f'{input_file} does not exist!')


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 6:
        print(
            'Usage: python3 move_files.py <input_dir> <output_dir> <name> <range_start> <range_end>')
        exit()
    input_dir = ''
    output_dir = ''
    name = ''
    range_start = 0
    range_end = 0

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    name = sys.argv[3]
    range_start = int(sys.argv[4])
    range_end = int(sys.argv[5])

    main(input_dir, output_dir, name, range_start, range_end)
