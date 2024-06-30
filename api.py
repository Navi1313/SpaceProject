from fastapi import FastAPI, Query
from pred import make_pred, model_path
import numpy as np

app = FastAPI()

@app.get('/')
def index_route():
    return {"health": "ok"}

@app.post('/predict')
def prediction(temperature, luminosity, radius, abs_mag):
    input_features = [[temperature, luminosity, radius, abs_mag]]
    pred_class, probs, classes = make_pred(model_path, input_features)
    
  
    # Convert numpy arrays to lists
    return {
         "Predicted_class": pred_class
    }
    
