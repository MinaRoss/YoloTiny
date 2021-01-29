如何训练模型？

    1. 从命令行开始 python run_console.py
    2. 交互界面开始 python run_ui.py

如何使用训练结果

    1. 可运行目录下的文件‘eval.py’
    
        python eval.py --path img_file_path --net_path(optional) net_file_path
        
        img_file_path：图片文件路径
        net_file_path：可指定网络文件路径，默认读取同目录下‘./model/yolo-tiny.pth’
    
    2.如果需要在另外的.py文件中调用，调用eval.py中的Eval函数即可，需要以下参数
    
        path：必须参数，为图片文件位置
        net_path：可选参数，为网络文件位置
