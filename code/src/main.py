from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

import re



app = FastAPI()

origins = [
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./models/modelBestTorch"
MAX_LEN = 128

print("Loading model...")
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH, from_transformers=True, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, max_length=MAX_LEN, padding="max_length")
print("Finished Loading")

def prettify_output(text):
  text = text.split(":")
  a = text[1].split(' ')
  a.remove(a[0])
  x = a.copy()
  x.remove(a[-1])
  x = ' '.join(x)
  feature = text[0] + ": " + x
  # print(feature)
  text.remove(text[0])
  # print(a)
  text[0] = a[-1]
  text[1] = re.sub(r"(?<!^)(?=[A-Z])", "\n", text[1])
  text[1] = text[1].replace(" \n", ": ", 1)
  text[1] = text[1].replace("\n", "\n\n", 1)

  scenario = ''.join(text)

  final_text = feature + "\n" + scenario

  return final_text


# Create the Data Model
class ModelInput(BaseModel):
    inputs: list

# Add endpoint for prediction
@app.post('/predict')
async def predict_endpoint(inputs: ModelInput):
    inputs = inputs.inputs #Getting inputs from the json
    inputs = list(map(lambda x: f"Generate test case for: {x}", inputs))
    inputs = tokenizer(inputs, return_tensors="pt", max_length=MAX_LEN, padding="max_length")
    outputs = model.generate(**inputs, max_new_tokens=MAX_LEN)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    try:
        outputs = list(map(lambda text: prettify_output(text), outputs))
    
    except Exception as e:
        pass

    return {
        'outputs': outputs
    }