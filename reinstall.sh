#!/bin/bash

echo "开始清理旧的wheel文件..."
cd dist && rm *.whl 
if [ $? -ne 0 ]; then  # 查看命令是否成功执行
    echo "清理wheel文件失败"
    exit 1
fi
cd ..

echo "开始移除旧的包安装信息..."
rm -rf /home/johnson/.local/lib/python3.10/site-packages/vllm_xft-0.4.2.0.dist-info
if [ $? -ne 0 ]; then
    echo "移除包安装信息失败"
    exit 1
fi

echo "开始构建新的wheel包..."
python3 setup.py bdist_wheel --verbose
if [ $? -ne 0 ]; then
    echo "构建wheel包失败"
    exit 1
fi

echo "开始安装新的wheel包..."
cd dist && pip install *.whl
if [ $? -ne 0 ]; then
    echo "安装wheel包失败"
    exit 1
fi

echo "所有操作完成"