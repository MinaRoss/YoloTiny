import json
import os
import datetime


def init_settings(fixed_config_path=r'./config.json'):
    if os.path.exists(fixed_config_path):
        print("[{}]已找到设置文件".format(datetime.datetime.now()))
    else:
        print("[{}]未找到设置文件，正在初始化原始设置...".format(datetime.datetime.now()))
        return format_missing_settings()
    try:
        with open(fixed_config_path, 'r') as f:
            settings = json.load(f)
            settings['anchors'] = dict([(int(item[0]), item[1]) for item in settings['anchors'].items()])
            settings['areas'] = calc_area(settings['anchors'])
            print("[{}]设置读取完成".format(datetime.datetime.now()))
        return settings
    except Exception as e:
        print("[{}]设置文件读取发生错误".format(datetime.datetime.now()))
        print("*--------------------------------------------*")
        print(e)
        print("*--------------------------------------------*")
        raise e


def format_missing_settings():
    fixed_format_settings = {
        "dataset_dir": "./dataset/train",
        "valid_dir": "./dataset/valid",
        "test_dir": "./dataset/test",
        "net_path": "./model/yolo-tiny.pth",
        "anchors": {
            "13": [[30, 61], [62, 45], [59, 119]],
            "26": [[116, 90], [156, 198], [373, 326]]
        },
        "epochs": 500,
        "batch_size": 32,
        "launch_mode": "train",
        "is_new": False
    }
    with open(r'./config.json', 'a+') as f:
        json.dump(fixed_format_settings, f)
        print("[{}]已初始化原始设置，设置文件已保存".format(datetime.datetime.now()))
    fixed_format_settings['anchors'] = dict([(int(item[0]), item[1]) for item in fixed_format_settings['anchors'].items()])
    fixed_format_settings['areas'] = calc_area(fixed_format_settings['anchors'])
    return fixed_format_settings


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
        print("* [{}]: {}".format(settings_key, settings[settings_key]))
    print("*---------------------------------------------*")
