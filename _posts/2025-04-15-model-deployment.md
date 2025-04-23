---
layout: post
author: Mark Hobbs
title: Deploying models using Flask
draft: True
---

This post explores the deployment of numerical and machine learning models in the cloud using Flask. By deploying models in the cloud and exposing their functionality via APIs, installation barriers are eliminated thus enabling broader adoption. The benefits of deploying models in the cloud include:

- Users can utilise sophisticated models without local software installation
- Computational resources can be scaled on demand
- Models and services can be easily connected together to form automated workflows
- Cross-platform compatibility
- Updates and improvements can be rolled out centrally

We will detail the process of exposing predictions using a pre-trained model. This is relatively computationally cheap but due to the computationally expensive nature of many models, we must think more carefully about handling long running requests. In most use cases a frontend will not be required, as the primary goal is to create an accessible API endpoint.

Docker... Gunicorn... Flask... Celery... UV...

- Cloud platforms (AWS, Google Cloud, Azure)
- Container services (Docker, Kubernetes)

### File structure

```bash
service/
├── app.py             # Flask app setup and routing
├── model.py           # Load and manage the model
├── services.py        # Logic or processing services (decouple from app.py)
├── utils.py           # Helper functions: e.g. file handling
├── pretrained.npz     # Model weights
run.py                 # Entry point
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
model.load(os.path.join("service", "pretrained.npz"))


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


### Running locally

It is now possible to serve the microservice locally by simply running `python run.py`. The user can test the application: 

```bash
curl -X POST http://server-address/predict -F "file=@input.csv"
```

To take the microservice from running locally to production-ready deployment...

## Deployment

### Docker

```docker
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Command to run the app using gunicorn
CMD ["gunicorn", "service.app:app", "--bind", "0.0.0.0:5000", "--workers=2"]
```

### Gunicorn configuration

Compute resources to use for online prediction

### Creating task queues

For models that take a long time to run, serving prediction directly from the Flask route can block the web server.

Supporting synchronous and/or asynchronous workflows

### Cloud-native parallelism

For trivially parallel tasks, such as... genetic algorithm... take advantage of scalable cloud infrastructure.

## Summary

By following the above practices, it is possible to create robust cloud-based APIs for numerical and machine learning models, making them accessible to users without the traditional barriers to adoption.