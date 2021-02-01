from utils import init_settings, print_settings, apply_new_settings
from yolo_tiny.trainer import Trainer
import datetime


class Launcher:
    def __init__(self, settings=None):
        self.settings = init_settings()
        self.check_new_settings(settings)

    def run(self):
        print("[{}]程序启动中...".format(datetime.datetime.now()))
        if self.settings['launch_mode'] == 'train':
            Trainer(self.settings['dataset_dir'],
                    self.settings['net_path'],
                    self.settings['epochs'],
                    self.settings['batch_size'],
                    self.settings['anchors'],
                    self.settings['areas'],
                    self.settings['is_new'],
                    self.settings['plot_interval'],
                    self.settings['log_dir'],
                    self.settings['save_loss_plot'],
                    self.settings['plot_pause'],
                    self.settings['plot_loss'])
            print("[{}]程序完成".format(datetime.datetime.now()))
        elif self.settings['launch_mode'] == 'test':
            raise NotImplementedError
        else:
            raise Exception("启动模式错误！")

    def check_new_settings(self, settings):
        if settings:
            self.settings = apply_new_settings(settings, self.settings)
            print("[{}]新设置已起效".format(datetime.datetime.now()))
        else:
            print("[{}]正在使用默认设置".format(datetime.datetime.now()))
        print_settings(self.settings)

