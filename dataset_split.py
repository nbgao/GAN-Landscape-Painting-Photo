import argparse
import os
import sys
import shutil
import random
from tqdm import tqdm
import time

def dataset_split(name, dataset_path, output_path, train_num, test_num):
    origin_path = os.path.join(dataset_path, name)
    train_path = os.path.join(output_path, 'train'+name)
    test_path = os.path.join(output_path, 'test'+name)

    file_list = os.listdir(origin_path)
    random.shuffle(file_list)
    print('orgin_path:', origin_path)
    print('files num:', len(file_list))
    # print(file_list[:10])
    print('train_path:', train_path)
    print('test_path:', test_path)
    
    train_file_list = file_list[:train_num]
    test_file_list = file_list[train_num: train_num+test_num]

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    
    for file in tqdm(train_file_list):
        src_path = os.path.join(origin_path, file)
        dst_path = os.path.join(train_path, file)
        shutil.copyfile(src_path, dst_path)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    for file in tqdm(test_file_list):
        src_path = os.path.join(origin_path, file)
        dst_path = os.path.join(test_path, file)
        shutil.copyfile(src_path, dst_path)
    

if __name__ == '__main__':
    time_start = time.time()

    parser = argparse.ArgumentParser(description='Dataset split')
    parser.add_argument('--dataset', '-dataset', type=str, default='Landscape', help='Dataset name')
    parser.add_argument('--trainA_num', '-trainA_num', type=int, default=2000, help='trainA image num')
    parser.add_argument('--testA_num', '-testA_num', type=int, default=100, help='testA image num')
    parser.add_argument('--trainB_num', '-trainB_num', type=int, default=2000, help='trainB image num')
    parser.add_argument('--testB_num', '-testB_num', type=int, default=100, help='testB image num')
    args = parser.parse_args()
    dataset = args.dataset
    train_paint_num = args.trainA_num
    test_paint_num = args.testA_num
    train_natural_num = args.trainB_num
    test_natural_num = args.testB_num
    print('dataset:', dataset)
    print('train paint image num:', train_paint_num)
    print('test paint image num:', test_paint_num)
    print('train natural image num:', train_natural_num)
    print('test natural image num:', test_natural_num)

    dataset_path = os.path.join('./Image_origin/', dataset)
    output_path = os.path.join('./U-GAT-IT/dataset/', dataset)

    dataset_split('A', dataset_path, output_path, train_paint_num, test_paint_num)
    dataset_split('B', dataset_path, output_path, train_natural_num, test_paint_num)

    time_end = time.time()
    print('Running time: {:.3f}s'.format(time_end-time_start))




