import requests

q = "The product I received is damaged, what now?"
resp = requests.post("http://127.0.0.1:8000/chat", json={"query": q, "top_k": 3})
print(resp.json())
