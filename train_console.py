from launcher import Launcher
import argparse


def extract_args():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--mode', nargs='?', type=str, default='train',
                        help='指定启动模式[train 训练模式]')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()
    runner = Launcher('console')
    runner.run()
