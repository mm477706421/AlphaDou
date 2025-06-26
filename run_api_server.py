import os
import argparse
from api import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AlphaDou API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    parser.add_argument('--gpu_device', type=str, default='', help='使用的GPU设备')
    args = parser.parse_args()

    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    # 启动Flask服务器
    app.run(host=args.host, port=args.port, debug=args.debug) 