import requests
import json
# The input text Can be enhanced to read from a file
input_text = ["money laundering", "fake login", "fraud risk"]
outputs = requests.post('http://localhost:8000/predict', json.dumps({'inputs': input_text})).json()["outputs"]

print("*********************")

for output in outputs:
    print(output)
    print("\n")
    print("*********************")
    print("\n")