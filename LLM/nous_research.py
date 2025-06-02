import requests

url = "https://inference-api.nousresearch.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {api-key}"
}
data = {
  "model": "Hermes-3-Llama-3.1-70B",
  "messages": [
    {
      "role": "user",
      "content": "{content}"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
