curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "xft",
  "prompt": "The president of the United States is",
  "max_tokens": 200,
  "temperature": 0,
  "stream": true
  }'