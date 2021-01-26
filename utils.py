import json
import os


def init_settings(fixed_config_path=r'./config.json'):
    if check_settings_file(fixed_config_path):
        print()
    else:
        print()
    try:
        with open(fixed_config_path, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        print()
        raise e


def check_settings_file(config_path):
    if not os.path.exists(config_path):
        print()
        return False
    else:
        print()
        return True
