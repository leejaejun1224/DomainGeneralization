

kitti_2015_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

kitti_2012_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

cityscapes_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

# source_dataset = 'kitti_2015'
source_dataset = 'flyingthing'
# source_dataset = 'cityscapes'
# source_dataset = 'driving_stereo'

# target_dataset = 'kitti_2015'
target_dataset = 'flyingthing'
# target_dataset = 'cityscapes'
# target_dataset = 'driving_stereo'

source_train = {}
source_train['type'] = f'{source_dataset}'
source_train['data_root'] = f'~/dataset/{source_dataset}'
source_train['file_list'] = f'./filenames/source/{source_dataset}_train.txt'
source_train['pipeline'] = cityscapes_pipeline

source_val = {}
source_val['type'] = f'{source_dataset}'
source_val['data_root'] = f'~/dataset/{source_dataset}'
source_val['file_list'] = f'./filenames/source/{source_dataset}_val.txt'
source_val['pipeline'] = cityscapes_pipeline

source_test = {}
source_test['type'] = f'{source_dataset}'
source_test['data_root'] = f'~/dataset/{source_dataset}'
source_test['file_list'] = f'./filenames/source/{source_dataset}_test.txt'
source_test['pipeline'] = cityscapes_pipeline

target_train = {}
target_train['type'] = f'{target_dataset}'
target_train['data_root'] = f'~/dataset/{target_dataset}'
target_train['file_list'] = f'./filenames/target/{target_dataset}_train.txt'
target_train['pipeline'] = kitti_2015_pipeline

target_val = {}
target_val['type'] = f'{target_dataset}'
target_val['data_root'] = f'~/dataset/{target_dataset}'
target_val['file_list'] = f'./filenames/target/{target_dataset}_val.txt'
target_val['pipeline'] = kitti_2015_pipeline

target_test = {}
target_test['type'] = f'{target_dataset}'
target_test['data_root'] = f'~/dataset/{target_dataset}'
target_test['file_list'] = f'./filenames/target/{target_dataset}_test.txt'
target_test['pipeline'] = kitti_2015_pipeline

dataset = dict(
    train = dict(
        type='UDAdataset',
        source=source_train,
        target=target_train
    ),
    val = dict(
        type='UDAdataset',
        source=source_val,
        target=target_val
    ),
    test = dict(
        type='UDAdataset',
        source=source_train,
        target=target_train
        )
    )