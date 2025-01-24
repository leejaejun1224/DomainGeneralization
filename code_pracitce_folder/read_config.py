import importlib.util

def load_config(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# config.py 파일 경로
config_path = './config.py'

# 모듈 이름은 자유롭게 설정 가능
config = load_config('custom_name', config_path)

# source 사용
print(config.source)
