


depth_model = dict(
    name = 'Fast_ACVNet_plus',
    # name = 'Fast_ACVNet_plus_refine',
    # name = 'Fast_ACVNet',
    maxdisp = 192,
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
    adjust_lr = "350,500,1000:10"
)

# name_dataset = 'kitti2015_to_kitti2012'
name_dataset = 'cityscapes_to_kitti2015'
# name_dataset = 'kitti2015_to_cityscapes'

data = dict(
    train = dict(
        warm_up = 0,
        epoch = 10,
        batch_size = 2,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
        val_interval = 2,
        save_interval = 2
    ),
    test = dict(
        batch_size = 1,
        num_workers = 1,
        shuffle = False,
        pin_memory = True
    )
)

lora = dict(
    student_lora = False,
    teacher_lora = False,
)