import json
import requests

data = {"name": "iPhone and Apple",
        "tags": ['book', 'towel'],
        "item_id": 23}

r = requests.post("http://127.0.0.1:8000/items/",
                  data=json.dumps(data))

print(r.json())