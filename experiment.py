import importlib.util



def prepare_cfg(arg):
    
    dataset_name = 'dataset_config'
    dataset_path = arg.dataset_config
    dataset_spec = importlib.util.spec_from_file_location(dataset_name, dataset_path)
    dataset_config = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_config)


    uda_name = 'uda_config'
    uda_path = arg.uda_config
    uda_spec = importlib.util.spec_from_file_location(uda_name, uda_path)
    uda_config = importlib.util.module_from_spec(uda_spec)
    uda_spec.loader.exec_module(uda_config)
    
    # 이러면 train, val, test를 다 나눠야 하는데 이게 맞나...
    cfg = dict(
        data = dict(
            src = dataset_config.data['train']['source']['type'], 
            tgt = dataset_config.data['train']['target']['type'],
            src_root = dataset_config.data['train']['source']['data_root'],
            tgt_root = dataset_config.data['train']['target']['data_root'],
            src_filelist = dataset_config.data['train']['source']['file_list'],
            tgt_filelist = dataset_config.data['train']['target']['file_list']
        ),
        batch_size = dataset_config.batch_size,
        num_workers = dataset_config.num_workers,

        model = dict(
            name = uda_config.depth_model['name'],
            maxdisp = uda_config.depth_model['maxdisp'],
            att_weights_only = uda_config.depth_model['att_weights_only']
        ),
        optimizer = dict(
            optimizer = uda_config.optimizer['optimizer'],
            lr = uda_config.optimizer['lr']
        ),
        uda = dict(
            threshold = uda_config.uda['threshold'],
            alpha = uda_config.uda['alpha']
        )
    )



    return cfg