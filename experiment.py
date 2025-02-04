import importlib.util


def load_config(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def prepare_cfg(arg):
    dataset_config = load_config('dataset_config', arg.dataset_config)
    uda_config = load_config('uda_config', arg.uda_config)
    
    train_source = dataset_config.dataset['train']['source']
    train_target = dataset_config.dataset['train']['target']
    
    cfg = dict(
        dataset = dict(
            src = train_source['type'],
            tgt = train_target['type'], 
            src_root = train_source['data_root'],
            tgt_root = train_target['data_root'],
            src_filelist = train_source['file_list'],
            tgt_filelist = train_target['file_list']
        ),
        batch_size = uda_config.data['train']['batch_size'],
        num_workers = uda_config.data['train']['num_workers'],
        model = uda_config.depth_model['name'],
        optimizer = uda_config.optimizer['optimizer'],
        uda = uda_config.uda,
        save_interval = uda_config.data['train']['save_interval']
    )

    return cfg
