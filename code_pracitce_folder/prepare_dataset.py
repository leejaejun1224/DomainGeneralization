# read dataset folder structure and write filename.txt
import os


modes = ['train', 'val', 'test']
for mode in modes:
    left_dir = '/home/jaejun/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/' + mode
    right_dir = '/home/jaejun/dataset/cityscapes/rightImg8bit_trainvaltest/rightImg8bit/' + mode
    disparity_dir = '/home/jaejun/dataset/cityscapes/disparity_trainvaltest/disparity/' + mode

    left = '/'.join(left_dir.split('/')[-3:])
    right = '/'.join(right_dir.split('/')[-3:])
    disparity = '/'.join(disparity_dir.split('/')[-3:])

    test_left_list = []
    for city in os.listdir(left_dir):
        for file in os.listdir(os.path.join(left_dir, city)):
            file_name = os.path.join(left, city, file)
            test_left_list.append(file_name)

    test_right_list = []
    for city in os.listdir(right_dir):
        for file in os.listdir(os.path.join(right_dir, city)):
            file_name = os.path.join(right, city, file)
            test_right_list.append(file_name)

    test_disparity_list = []
    for city in os.listdir(disparity_dir):
        for file in os.listdir(os.path.join(disparity_dir, city)):
            file_name = os.path.join(disparity, city, file)
            test_disparity_list.append(file_name)

    with open('./filenames/target/cityscapes_' + mode + '.txt', 'w') as f:
        for i in range(len(test_left_list)):
            f.write(test_left_list[i] + ' ' + test_right_list[i] + ' ' + test_disparity_list[i] + '\n')


