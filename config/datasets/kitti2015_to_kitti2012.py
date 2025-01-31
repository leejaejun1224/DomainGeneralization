
batch_size = 1
num_workers = 2


kitti_2015_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

kitti_2012_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

cityscapes_pipeline = [
    dict(type='Resize', img_scale=(1248, 384))
]

data = dict(
    train = dict(
        type='UDAdataset',
        source=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015',
            img_dir='image_2',
            ann_dir='2015 disparity gt 저장된 곳',
            file_list='./filenames/source/kitti15.txt',
            pipeline=kitti_2015_pipeline
        ),
        target=dict(
            type='kitti2015',
            data_root='/home/jaejun/dataset/kitti/kitti2015',
            img_dir='image_2',
            ann_dir='2012 disparity gt 저장된 곳',
            file_list='./filenames/target/kitti15.txt',
            pipeline=kitti_2012_pipeline
        )
    ),
    val = dict(
        
    ),
    test = dict(
        
    )
)
