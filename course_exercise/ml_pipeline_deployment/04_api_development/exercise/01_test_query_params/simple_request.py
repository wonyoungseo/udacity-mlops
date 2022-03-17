import json
import requests

# data = {
#         "path": 'test',
#         "query": 5,
#         "body": "hello world"
#     }


data = {
        "path": 1,
        "query": 5,
        "body": 9
    }

r = requests.post("http://127.0.0.1:8000/",
                  data=json.dumps(data))

print(r.json())