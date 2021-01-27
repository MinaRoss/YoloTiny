import json
import os
import argparse
import datetime


def init_settings(launch_mode, fixed_config_path=r'./config.json'):
    if check_settings_file(fixed_config_path):
        print("[{}][{}]已找到设置文件".format(datetime.datetime.now(), launch_mode))
    else:
        print("[{}][{}]未找到设置文件，正在初始化原始设置...".format(datetime.datetime.now(), launch_mode))
    try:
        with open(fixed_config_path, 'r') as f:
            settings = json.load(f)
            settings['anchors'] = dict([(int(item[0]), item[1]) for item in settings['anchors'].items()])
            settings['areas'] = calc_area(settings['anchors'])
            print("[{}][{}]设置读取完成".format(datetime.datetime.now(), launch_mode))
        return settings
    except Exception as e:
        print("[{}][{}]设置文件读取发生错误")
        print("*--------------------------------------------*")
        print(e)
        print("*--------------------------------------------*")
        raise e


def check_settings_file(config_path):
    if not os.path.exists(config_path):
        print()
        return False
    else:
        print()
        return True


def check_missing_settings():
    pass


def calc_area(anchors):
    areas = {}
    for feature_size in anchors:
        areas[feature_size] = [x * y for x, y in anchors[feature_size]]
    return areas


def apply_new_settings(changed_settings, original_settings):
    for settings_key in changed_settings:
        original_settings[settings_key] = changed_settings[settings_key]
    return original_settings


def print_settings(settings):
    print("*---------------------------------------------*")
    for settings_key in settings:
        print("[{}]: {}".format(settings_key, settings[settings_key]))
    print("*---------------------------------------------*")


def extract_settings():
    pass
