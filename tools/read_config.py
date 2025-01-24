import importlib.util

## 모듈 이름은 우리가 설정,path는 해당 config.py 이거를 말함.
def load_config(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

