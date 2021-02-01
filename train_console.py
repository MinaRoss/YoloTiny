from launcher import Launcher
import argparse
import json


def cvt_args2dict(args):
    filled_settings = {}
    for args_key, args_value in args._get_kwargs():
        if args_value:
            if args_key == 'anchors':
                args_value = json.loads(args_value)
            filled_settings[args_key] = args_value
    if len(filled_settings) == 0:
        return None
    return filled_settings


if __name__ == '__main__':
    args_dict = {'launch_mode': ['--launch_mode', None, str, '指定启动模式'],
                 'dataset_dir': ['--dataset_dir', None, str, '训练数据目录'],
                 'valid_dir': ['--valid_dir', None, str, '验证数据目录'],
                 'test_dir': ['--test_dir', None, str, '测试数据目录'],
                 'net_path': ['--net_path', None, str, '网络保存路径'],
                 'anchors': ['--anchors', None, str, '目标参考框'],
                 'epochs': ['--epochs', None, int, '训练轮次'],
                 'batch_size': ['--batch_size', None, int, '训练批次大小'],
                 'is_new': ['--is_new', False, bool, '指定是否重头训练'],
                 'log_dir': ['--log_dir', None, str, 'Loss画图保存地址'],
                 'plot_interval': ['--plot_interval', None, int, 'Loss画图的间隔'],
                 'save_loss_plot': ['--save_loss_plot', False, bool, '是否保存Loss画图'],
                 'plot_pause': ['--plot_pause', None, float, '画图暂定时长'],
                 'plot_loss': ['--plot_loss', False, bool, '是否可视化Loss']}
    parser = argparse.ArgumentParser(description='Hyperparams')
    for args_key, args_list in args_dict.items():
        parser.add_argument(args_list[0], nargs='?', default=args_list[1], type=args_list[2], help=args_list[3])
    args = parser.parse_args()
    filled_settings = cvt_args2dict(args)
    runner = Launcher(filled_settings)
    runner.run()
