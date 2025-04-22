import requests

r = requests.post("http://localhost:8000/load_model", json={"path":"./server_model"})
print(r.json())
