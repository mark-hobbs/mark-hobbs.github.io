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

### Pretrained weights

### Docker