

kitti_2015_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

kitti_2012_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

cityscapes_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

dataset = dict(
    train = dict(
        type='UDAdataset',
        source=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/training',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2015 disparity gt 저장된 곳',
            file_list='./filenames/source/kitti15_train.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/training',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2012 disparity gt 저장된 곳',
            file_list='./filenames/target/kitti15_train.txt',
            pipeline=kitti_2012_pipeline
        )
    ),
    val = dict(
        type='UDAdataset',
        source=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/training',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2015 disparity gt 저장된 곳',
            file_list='./filenames/source/kitti15_val.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/training',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2012 disparity gt 저장된 곳',
            file_list='./filenames/target/kitti15_val.txt',
            pipeline=kitti_2012_pipeline
        )
    ),
    test = dict(
                type='UDAdataset',
        source=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/testing',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2015 disparity gt 저장된 곳',
            file_list='./filenames/source/kitti15_test.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015/testing',
            left_img_dir='image_2',
            right_img_dir='image_3',
            ann_dir='2012 disparity gt 저장된 곳',
            file_list='./filenames/target/kitti15_test.txt',
            pipeline=kitti_2012_pipeline
        )
    )
)
