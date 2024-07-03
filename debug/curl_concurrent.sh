#!/bin/bash

# 定义请求次数
REQUEST_COUNT=10

for ((i=1; i<=REQUEST_COUNT; i++)); do
  curl -sS http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "xft",
      "prompt": "The president of the United States is",
      "max_tokens": 10,
      "temperature": 0,
      "stream": false
    }' &
  sleep 0.5
   
done

# 等待所有后台进程完成
wait