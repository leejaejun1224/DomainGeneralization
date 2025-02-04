


depth_model = dict(
    name = 'Fast_ACVNet',
    maxdisp = 192,
    att_weights_only = False
)

uda = dict(
    threshold = 0.2,
    alpha = 0.99
)



optimizer = dict(
    optimizer = "Adam",
    lr = 1e-4
)



name_dataset = 'kitti2015_to_kitti2012'

epoch = 100