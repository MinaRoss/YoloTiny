from utils import init_settings, print_settings, apply_new_settings
from yolo_tiny.trainer import Trainer
import datetime


class Launcher:
    def __init__(self, launch_mode='console'):
        print("[{}]运行模式 [{}]".format(datetime.datetime.now(), launch_mode))
        print("[{}][{}]初始化中...".format(datetime.datetime.now(), launch_mode))
        self.settings = init_settings(launch_mode)
        self.launch_mode = launch_mode

    def run(self, settings=None):
        print("[{}][{}]程序启动中...".format(datetime.datetime.now(), self.launch_mode))
        if self.launch_mode == 'ui':
            self.launch_ui()
        elif self.launch_mode == 'console':
            self.launch_console(settings)
        else:
            raise Exception("启动模式错误！")

    def launch_ui(self):
        raise Exception("模式还未开放")

    def launch_console(self, settings):
        if settings:
            self.settings = apply_new_settings(settings, self.settings)
            print("[{}][{}]新设置已起效".format(datetime.datetime.now(), self.launch_mode))
        else:
            print("[{}][{}]正在使用默认设置".format(datetime.datetime.now(), self.launch_mode))
        print_settings(self.settings)

        Trainer(self.settings['dataset_dir'],
                self.settings['net_path'],
                self.launch_mode,
                self.settings['epochs'],
                self.settings['batch_size'],
                self.settings['anchors'],
                self.settings['areas'])
