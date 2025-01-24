import importlib.util



def prepare_cfg(arg):
    
    module_name = 'dataset_config'
    file_path = arg.dataset_config
    
    dataset_spec = importlib.util.spec_from_file_location(module_name, file_path)
    dataset_config = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_config)
    
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
        num_workers = dataset_config.num_workers
        
    )
    return cfg
