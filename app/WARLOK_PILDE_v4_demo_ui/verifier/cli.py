
import requests

API = "http://127.0.0.1:8000"

h = input("Enter hash: ")
res = requests.post(f"{API}/validate", json={"hash": h})
print(res.json())
