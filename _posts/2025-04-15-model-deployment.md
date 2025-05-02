---
layout: post
author: Mark Hobbs
title: Deploying models using Flask
draft: True
---

This post explores the deployment of numerical and machine learning models in the cloud using Flask. By deploying models in the cloud and exposing their functionality via APIs, installation barriers are eliminated thus enabling broader adoption. The benefits of deploying models in the cloud include:

- **No local setup required:** Users can access complex models via a simple HTTP request, with no need to install dependencies or configure environments.
- **Seamless integration:** Easily connect models with other services to build automated workflows.
- **Cross-platform accessibility:** APIs work consistently across devices and operating systems.
- **Elastic scalability:** Cloud infrastructure allows computational resources to be scaled up or down based on demand.
- **Centralised updates:** Models can be improved or retrained, and changes take effect immediately for all users without requiring reinstallation.

We will detail the process of exposing predictions using a pre-trained model. This is relatively computationally cheap but due to the computationally expensive nature of many models, we must think more carefully about handling long running requests. In most use cases a frontend will not be required, as the primary goal is to create an accessible API endpoint.

We refer to the framework that delivers this model functionality as a *service* (or *microservice*). While definitions of microservices vary, the core idea remains simple: a service provides a well-defined capability that accepts input and returns output - cleanly and reliably.

## Building the service

### Functionality

The service exposes a pre-trained regression model that predicts the value of a continuous target variable $Y$ from four input features $(X_1, X_2, X_3, X_4)$. Once deployed, the model can be accessed via a lightweight API call, returning predictions in real time with minimal computational overhead.

### Design

The service adopts a modular design that cleanly separates concerns and enhances maintainability. All the code can be found [here](https://github.com/mark-hobbs/articles/tree/main/model-deployment).

```bash
service/
├── app.py
├── model.py
├── services.py
├── utils.py
├── pretrained-model.pkl
run.py
```

### `app.py`

The `app` module sets up the Flask application and defines the API endpoints. It serves as the interface between the user and the underlying logic, routing incoming requests to the appropriate service functions. While this example demonstrates a single endpoint, `/predict`, other endpoints, such as `/train`, can easily be added to expand the functionality of the application.

```python
from flask import Flask, request

from . import services

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests via POST and return the result as JSON
    """
    return services.predict(request.json)
```

### `services.py`

The `services` module contains the core business logic or processing functions. This decouples the application logic from the API layer, making the system more testable and easier to extend. 

Custom functionality will likely be required to convert the input data - whether passed as query parameters, uploaded files, or JSON bodies - into a suitable format. Inputs might range from a single parameter to a large 3D mesh. This is where `services.py` acts as the bridge between raw requests and meaningful model inputs.

```python
import os

from flask import jsonify

from .model import GPR
from .utils import json_to_ndarray

model = GPR()
model.load(os.path.join("service", "pretrained-model.pkl"))


def predict(input):
    """
    Predict... using a pre-trained model

    Args:
        - input

    Returns:
        - JSON response
    """
    try:
        mean, var = model.predict(json_to_ndarray(input))
        return jsonify({"mean": float(mean[0, 0]), "variance": float(var[0, 0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### `model.py`

A generic `Model` base class that abstracts the complexity of the numerical or machine learning model behind a consistent interface. 

This class is intentionally minimal and problem-agnostic. While not all models will require methods like `train` or `save`, these are included as common entry points to encourage consistency across different implementations. Subclasses should override only the methods relevant to their use case.

The `GPR` class is a specific implementation of the `Model` interface for Gaussian Process Regression (GPR) using the `GPy` library. It assumes that the model has been pre-trained and saved to disk. The load method deserialises a trained `GPRegression` instance, while `predict` provides access to its prediction functionality. This class does not support training or saving, as model fitting occurs offline.

```python
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all models
    """

    def __init__(self):
        self.trained = False

    @abstractmethod
    def predict(self, X):
        pass

    def fit(self, X, y):
        raise NotImplementedError("This model type does not support training")

    def save(self):
        raise NotImplementedError("This model type does not support saving")

    def load(self):
        raise NotImplementedError("This model type does not support loading")


import pickle


class GPR(Model):
    """
    Pre-trained Gaussian Process Regression model

    Attributes
    ----------
    model : GPRegression
        An instance of a trained GPy.models.gp_regression.GPRegression model
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def predict(self, X):
        return self.model.predict(X)

    def load(self, file):
        with open(file, "rb") as f:
            self.model = pickle.load(f)
            self.trained = True
```

### `utils.py`

Hosts helper functions used across the service, such as file handling and data preprocessing. This avoids duplication and keeps general utilities and tools out of the main service code.

```python
def json_to_ndarray(data):
    """
    Convert JSON data to a NumPy array.
    """
    import numpy as np

    return np.array([[data["x1"], data["x2"], data["x3"], data["x4"]]])
```

### `pretrained-model.pkl`

Contains pre-trained model weights that are loaded by the `Model` instance during initialisation. In this case, the training process occurs offline, and the trained model is serialised to disk. This approach enables the service to make predictions immediately at startup without requiring re-training, and ensures that potentially sensitive training data remains separate.

### `run.py`

The main entry point to the application, typically used to start the Flask server. Keeping this separate allows for easy deployment and testing.

```python
from service.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
```

### Running locally

To run the microservice locally, execute:

```bash
python run.py
``` 

You can then test the application by sending a request to the `/predict` endpoint: 

```bash
curl -X POST http://localhost:5001/predict  -H "Content-Type: application/json" -d @input.json
```

Where the input file is structured as follows:

```json
{
    "x1": 0.51,
    "x2": 0.76,
    "x3": 0.94,
    "x4": 0.72
  }
```

The server returns the prediction results in a JSON format.

```json
{
  "mean": -0.4507921664044261,
  "variance": 2.2272495243669255e-06
}
```

## Deploying in a production environment

While the service now runs locally, making it available to others requires deployment to a production environment. Thanks to containerisation tools like [Docker](https://www.docker.com/) and modern cloud platforms, this process is now straightforward. It is not the purpose of this article to explain this process in detail but the typical steps are as follows:

- Containerise the service using Docker to encapsulate the environment and dependencies
- Serve the application using a production-grade WSGI server such as Gunicorn
- Deploy the container to a scalable cloud platform (e.g. AWS, Azure, GCP or a Kubernetes cluster)
- Expose the service via a stable endpoint (e.g. `https://example.com/model/predict`)

A key advantage of deploying in the cloud is the ability to scale automatically in both directions - vertically, by allocating more powerful compute resources, and horizontally, by adding multiple independent instances that can run concurrently. This makes a cloud-based approach particularly well-suited to handling embarrassingly parallel workloads, such as those encountered in design of experiments or population-based optimisation methods, where many independent model evaluations can be run simultaneously with minimal overhead.

## Summary

By following the above practices, it is possible to create robust cloud-based APIs for numerical and machine learning models, making them accessible to users without the traditional barriers to adoption.  While this post uses a simple regression model for illustration, the architecture can easily scale to more complex or compute-intensive models with features like asynchronous processing, job queues and scheduling. This approach also enables seamless integration between models, paving the way for fully automated end-to-end workflows.

<!-- ### Creating task queues

For models that take a long time to run, serving prediction directly from the Flask route can block the web server.

Supporting synchronous and/or asynchronous workflows

### Cloud-native parallelism (horizontal cloud bursting)

Kubernetes + Horizontal Pod Autoscaling

For trivially parallel tasks, such as... genetic algorithm... take advantage of scalable cloud infrastructure.

Kubernetes... Horizontal Pod Autoscaling. Horizontal scaling means that the response is to deploy more Pods. This is different from *vertical* scaling, which for Kubernetes would mean assigning more resources (for example: memory or CPU) to the Pods that are already running for the workload. -->