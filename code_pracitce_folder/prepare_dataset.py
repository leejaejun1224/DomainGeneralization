# read dataset folder structure and write filename.txt
import os


cities = ['seoul', 'tokyo', 'beijing']


dir = './code_pracitce_folder/train/disparity/'


train_filename = 'cityscapes_train.txt'
val_filename = 'cityscapes_val.txt'
test_filename = 'cityscapes_test.txt'


for city in os.listdir(dir):
    if city in cities:
        with open(train_filename, 'a') as f:
            f.write(city + 'train\n')

