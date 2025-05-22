---
layout: post
author: Mark Hobbs
title: A universal interface...
draft: True
---

Integrating models, ranging from simple analytical models to complex numerical simulations, into larger workflows is often hindered by technical complexity. In this post, we motivate the need for a universal interface to wrap models and encapsulate them as modular building blocks, before exploring the design of a simple and consistent abstraction.

## Motivation

A model $F(\textbf{X})$ is a function that takes a set of input parameters $\textbf{X}$ and returns an output $y$. The model might be a computationally expensive numerical simulation or... 


- FEA, CFD, etc
- Outer-loop applications: design space exploration, optimisation, uncertainty quantification
- Chaining models together
- Provide a consistent high level abstraction that hides the complexity of models, only exposing their core functionality
- Call this high level abstraction in client side workflows
- Problem decomposition: how do you take a complicated problem/system and break it down into discrete units that can be built independently?
- Separating concerns
- Language agnostic

I am in the process of writing a similar post on [model calibration](https://mark-hobbs.github.io/posts/model-calibration).

## Design

The goal is to design a generic `Model` base class that abstracts the complexity of the numerical or machine learning model behind a consistent interface.

```python
from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def __call__(self, X) -> y:
        """
        Evaluate the model on input X
        """
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def forward(self, X) -> y:
        """
        Alias for __call__, can override for clarity
        """
        pass

    def fitness(self):
        """
        Objective function
        """
        pass
```

## Usage

By wrapping a computationally expensive peridynamic model that simulates the fracture behaviour of a three-point beam in bending... we can integrate the model into different workflows with ease. For example, here we will demonstrate the calibration of the constitutive model parameters ($\alpha$, $k$) to minimise the discrepancy between the model predictions and experimental data.

```python
class Beam(Model):

    import pypd

    def __init__(self):
        super().__init__()
        self.simulation = pypd.simulation(n_time_steps=100000, damping=0)

    def __call__(self, x):
        model = setup_problem(x)
        self.simulation.run(model)

    def get_input_size(self) -> int:
        """
        alpha and k
        """
        return 2

    def forward(self):
        self.simulation.run(self.model)

    def fitness(self):
        pass


def setup_problem(k, alpha):
    x = build_particle_coordinates(dx, n_div_x, n_div_y)
    flag, unit_vector = build_boundary_conditions(x)  # TODO: not needed

    material = pypd.Material(name="quasi-brittle", E=37e9, Gf=143.2, density=2346, ft=3.9e6)
    bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=0)
    particles = pypd.ParticleSet(x, dx, bc, material)

    radius = 25 * mm_to_m
    penetrators = []
    penetrators.append(
        pypd.Penetrator(
            np.array([0.5 * length, depth + radius - dx]),
            np.array([0, 1]),
            np.array([0, -0.4 * mm_to_m]),
            radius,
            particles,
            name="Penetrator",
            plot=False,
        )
    )
    penetrators.append(
        pypd.Penetrator(
            np.array([0.5 * depth, -radius]),
            np.array([0, 0]),
            np.array([0, 0]),
            radius,
            particles,
            name="Support - left",
            plot=False,
        )
    )
    penetrators.append(
        pypd.Penetrator(
            np.array([3 * depth, -radius]),
            np.array([0, 0]),
            np.array([0, 0]),
            radius,
            particles,
            name="Support - right",
            plot=False,
        )
    )

    observations = []
    observations.append(
        pypd.Observation(
            np.array([77.5 * mm_to_m, 0]), particles, period=1, name="CMOD - left"
        )
    )
    observations.append(
        pypd.Observation(
            np.array([97.5 * mm_to_m, 0]), particles, period=1, name="CMOD - right"
        )
    )

    bonds = pypd.BondSet(particles),
                         constitutive_law=pypd.NonLinear, 
                         constitutive_law_params={'alpha': alpha, 'k': k},
                         surface_correction=True, 
                         notch=notch)
    model = pypd.Model(particles, 
                       bonds, 
                       penetrators, 
                       observations)
```

```python

from scipy.optimize import minimize


beam = Beam()
result = minimize(beam, x0=[alpha, k], method='Nelder-Mead', options={'maxiter': 100})

```

## Microservices inspired

The above concept can be taken to the next level by adopting a microservices inspired workflow where a Docker image is built of the model... 

Clean separation between model logic and serving logic...


- Client side: control and orchestration... 
- Server side: model execution...

Whilst thinking about the above ideas... `um-bridge`... [https://github.com/um-bridge](https://github.com/um-bridge)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class Input(BaseModel):
    X: list[float]

def serve_model(model: Model):
    @app.post("/evaluate")
    def evaluate(input_data: Input):
        try:
            X = np.array(input_data.X).reshape(1, -1)
            y = model(X)
            return {"y": y.tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/meta")
    def meta():
        return {
            "input_size": model.get_input_size(),
            "output_size": model.get_output_size()
        }
```

### Example

```python
class ToyModel(Model):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X ** 2

    def get_input_size(self):
        return 1

    def get_output_size(self):
        return 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.__call__(X)
```

```python
if __name__ == "__main__":
    import uvicorn
    toy_model = ToyModel()
    serve_model(toy_model)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```