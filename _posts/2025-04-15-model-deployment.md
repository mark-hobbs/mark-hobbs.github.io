---
layout: post
author: Mark Hobbs
title: Deploying models using Flask
draft: True
---

This post explores the deployment of numerical and machine learning models in the cloud using Flask. By deploying models in the cloud and exposing their functionality via APIs... easily accessible, scalable and interconnected. In most use cases a frontend will not be required...

Due to the computationally expensive nature of many models... we must think more carefully about handling long running requests.

Docker... Gunicorn... Flask... Celery... UV...

### File structure

```bash
service/
├── __init__.py
├── app.py             # Entrypoint for the Flask app
├── routes.py          # API endpoints
├── services.py        # Core logic: inference, validation
├── model.py           # Load and manage the model
├── utilities.py       # Helper functions: e.g. file handling
├── config.py          # (Optional) Configuration settings
```

### Model

`model.py`

```python
class Model:

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.trained = False

    def train(self):
        return NotImplementedError

    def predict(self):
        return NotImplementedError

    def save(self):
        return NotImplementedError

    def load(self):
        return NotImplementedError
```

### Services

Core logic: inference, validation

```python
import os

from flask import jsonify

from .model import Model

model = Model()
model.load(os.path.join("model", "pretrained.npz"))


def predict(input):
    """
    Predict... using a pre-trained model
    
    Args:
        - input

    Returns:
        - JSON response
    """
    try:
        prediction = model.predict(input)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### Routes

Define the API endpoints.

```python
from flask import request, jsonify


from .services import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    return predict(input)
```

### Pretrained weights

### Docker