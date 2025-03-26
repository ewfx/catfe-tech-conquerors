import requests
import json



with open("./contexts.txt", "r") as file:
    input_text = file.read().splitlines()
    print("Contexts: ", input_text)

# input_text = ["money laundering", "fake login", "fraud risk"]
outputs = requests.post('http://localhost:8000/predict', json.dumps({'inputs': input_text}))
outputs = outputs.json()["outputs"]

print("*********************")

for output in outputs:
    print(output)
    print("\n")
    print("*********************")
    print("\n")