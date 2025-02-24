


depth_model = dict(
    name = 'Fast_ACVNet',
    maxdisp = 192,
    att_weights_only = False
)

uda = dict(
    threshold = 0.5,
    alpha = 0.99,
    train_source_only = True
)

optimizer = dict(
    optimizer = "Adam",
    lr = 1e-4
)

# name_dataset = 'kitti2015_to_kitti2012'
name_dataset = 'cityscapes_to_kitti2015'
# name_dataset = 'kitti2015_to_cityscapes'

data = dict(
    train = dict(
        epoch = 100,
        batch_size = 2,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
        val_interval = 1,
        save_interval = 1
    ),
    test = dict(
        batch_size = 1,
        num_workers = 2,
        shuffle = False,
        pin_memory = True
    )
)