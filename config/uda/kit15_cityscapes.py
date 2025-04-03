


depth_model = dict(
    name = 'Fast_ACVNet_plus',
    # name = 'Fast_ACVNet',
    maxdisp = 256,
    att_weights_only = False
)

uda = dict(
    train_threshold = 0.65,
    val_threshold = 0.4,
    alpha = 0.99,
    train_source_only = True
)

optimizer = dict(
    optimizer = "Adam",
    lr = 1e-4,
    adjust_lr = "250,400:10"
)

# name_dataset = 'kitti2015_to_kitti2012'
name_dataset = 'cityscapes_to_kitti2015'
# name_dataset = 'kitti2015_to_cityscapes'

data = dict(
    train = dict(
        epoch = 400,
        batch_size = 2,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
        val_interval = 50,
        save_interval = 50
    ),
    test = dict(
        batch_size = 2,
        num_workers = 2,
        shuffle = False,
        pin_memory = True
    )
)