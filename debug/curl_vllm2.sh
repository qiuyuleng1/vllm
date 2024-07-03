curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "xft",
  "prompt": "I have an apple",
  "max_tokens": 10,
  "temperature": 0,
  "stream": true
  }'

  # I have an apple 