

kitti_2015_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

kitti_2012_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

cityscapes_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

source_dataset = 'cityscapes'
# target_dataset = 'kitti_2015'
target_dataset = 'kitti_2015'


cityscapes_train = {}
cityscapes_train['data_root'] = '/home/jaejun/dataset/cityscapes'
cityscapes_train['file_list'] = './filenames/source/{source_dataset}_train.txt'
cityscapes_train['pipeline'] = cityscapes_pipeline

cityscapes_val = {}
cityscapes_val['data_root'] = '/home/jaejun/dataset/cityscapes'
cityscapes_val['file_list'] = './filenames/source/{source_dataset}_val.txt'
cityscapes_val['pipeline'] = cityscapes_pipeline

cityscapes_test = {}
cityscapes_test['data_root'] = '/home/jaejun/dataset/cityscapes'
cityscapes_test['file_list'] = './filenames/source/{source_dataset}_test.txt'
cityscapes_test['pipeline'] = cityscapes_pipeline

kitti_train = {}
kitti_train['data_root'] = '/home/jaejun/dataset/kitti'
kitti_train['file_list'] = './filenames/target/{target_dataset}_train.txt'
kitti_train['pipeline'] = kitti_2015_pipeline

kitti_val = {}
kitti_val['data_root'] = '/home/jaejun/dataset/kitti'
kitti_val['file_list'] = './filenames/target/{target_dataset}_val.txt'
kitti_val['pipeline'] = kitti_2015_pipeline

kitti_test = {}
kitti_test['data_root'] = '/home/jaejun/dataset/kitti'
kitti_test['file_list'] = './filenames/target/{target_dataset}_test.txt'
kitti_test['pipeline'] = kitti_2015_pipeline

dataset = dict(
    train = dict(
        type='UDAdataset',
        source=cityscapes_train,
        target=kitti_train
    ),
    val = dict(
        type='UDAdataset',
        source=cityscapes_val,
        target=kitti_val
    ),
    test = dict(
        type='UDAdataset',
        source=cityscapes_test,
        target=kitti_test
        )
    )

