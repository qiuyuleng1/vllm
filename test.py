import pickle

# 打开之前保存的pickle文件
with open('/home/johnson/qiuyu/vllm-xft/debug/pp3_output1.pkl', 'rb') as file:
    output = pickle.load(file)

# 现在 output 变量包含了之前保存的对象
print(output)