import importlib.util

def adjust_learning_rate(optimizer, epoch, start_lr, lr_per_epoch):
    splits = lr_per_epoch.split(':')
    assert len(splits) == 2

    downscale_epochs = [int(downscale_epoch) for downscale_epoch in splits[0].split(',')]
    downscale_rate = float(splits[1])

    lr = start_lr
    for downscale_epoch in downscale_epochs:
        if epoch >= downscale_epoch:
            lr /= downscale_rate
        else:
            break
    print("current learning rate is {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_error(pred_disp, gt_disp):
    mask = (gt_disp > 0) & (gt_disp < 192)
    return (pred_disp[mask] - gt_disp[mask]).abs().mean()


def load_config(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def prepare_cfg(arg, mode='train'):
    dataset_config = load_config('dataset_config', arg.dataset_config)
    uda_config = load_config('uda_config', arg.uda_config)
    
    # train_source = dataset_config.dataset['train']['source']
    # train_target = dataset_config.dataset['train']['target']
    
    if mode == 'train':
        train_source = dataset_config.dataset['train']['source']
        train_target = dataset_config.dataset['train']['target']
    else:
        train_source = dataset_config.dataset['test']['source']
        train_target = dataset_config.dataset['test']['target']
    
    cfg = dict(
        dataset = dict(
            src_root = train_source['data_root'],
            tgt_root = train_target['data_root'],
            src_filelist = train_source['file_list'],
            tgt_filelist = train_target['file_list']
        ),
        batch_size = uda_config.data['train']['batch_size'],
        num_workers = uda_config.data['train']['num_workers'],
        model = uda_config.depth_model['name'],
        maxdisp = uda_config.depth_model['maxdisp'],
        att_weights_only = uda_config.depth_model['att_weights_only'],
        optimizer = uda_config.optimizer['optimizer'],
        lr = uda_config.optimizer['lr'],
        uda = uda_config.uda,
        val_interval = uda_config.data['train']['val_interval'],
        save_interval = uda_config.data['train']['save_interval'],
        epoch = uda_config.data['train']['epoch'],
        train_batch_size = uda_config.data['train']['batch_size'],  
        train_num_workers = uda_config.data['train']['num_workers'],
        train_shuffle = uda_config.data['train']['shuffle'],
        train_pin_memory = uda_config.data['train']['pin_memory'],
        test_batch_size = uda_config.data['test']['batch_size'],
        test_num_workers = uda_config.data['test']['num_workers'],
        test_shuffle = uda_config.data['test']['shuffle'],
        test_pin_memory = uda_config.data['test']['pin_memory'],
        adjust_lr = uda_config.optimizer['adjust_lr']
    )

    return cfg
