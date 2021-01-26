from utils import init_settings
from yolo_tiny.trainer import Trainer


class Launcher:
    def __init__(self, launch_mode='ui'):
        self.settings = init_settings()
        self.launch_mode = launch_mode

    def run(self):
        if self.launch_mode == 'ui':
            self.launch_ui()
        elif self.launch_mode == 'console':
            self.launch_console()
        else:
            raise Exception("启动模式错误！")

    def launch_ui(self):
        pass

    def launch_console(self):
        pass
