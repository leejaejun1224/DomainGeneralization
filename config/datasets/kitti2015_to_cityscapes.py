

kitti_2015_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

kitti_2012_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

cityscapes_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

source_dataset = 'kitti_2015'
# target_dataset = 'kitti_2015'
target_dataset = 'cityscapes'

dataset = dict(
    train = dict(
        type='UDAdataset',
        source=dict(
            data_root=f'/home/jaejun/dataset/kitti',
            file_list=f'./filenames/source/{source_dataset}_train.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            data_root=f'/home/jaejun/dataset/cityscapes',
            file_list=f'./filenames/target/{target_dataset}_train.txt',
            pipeline=cityscapes_pipeline
        )
    ),
    val = dict(
        type='UDAdataset',
        source=dict(
            data_root=f'/home/jaejun/dataset/kitti',
            file_list=f'./filenames/source/{source_dataset}_val.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            data_root=f'/home/jaejun/dataset/cityscapes',
            file_list=f'./filenames/target/{target_dataset}_val.txt',
            pipeline=cityscapes_pipeline
        )
    ),
    test = dict(
        type='UDAdataset',
        source=dict(
            data_root=f'/home/jaejun/dataset/kitti',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2015 disparity gt 저장된 곳',
            file_list=f'./filenames/source/{source_dataset}_test.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            data_root=f'/home/jaejun/dataset/cityscapes',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2012 disparity gt 저장된 곳',
            file_list=f'./filenames/target/{target_dataset}_test.txt',
            pipeline=cityscapes_pipeline
        )
    )
)
