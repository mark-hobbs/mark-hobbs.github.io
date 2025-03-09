---
layout: post
author: Mark Hobbs
title: Calibrating computationally expensive models 
draft: True
---

This post is titled *"Calibrating computationally expensive models"*, but the ideas and software design patterns introduced are applicable to all *outer-loop* applications - including design space exploration, optimisation, uncertainty quantification and sensitivity analysis.

The calibration of computationally expensive numerical models to experimental data is a challenging task. Calibration is an iterative process, often requiring hundreds to thousands of repeat simulations to minimise the discrepancy between model predictions and experimental observations. If the model runtime is of the order of hours to days, then the calibration task becomes computationally impracticable.

To address this limitation, surrogate modelling approaches have emerged as effective alternatives, where the high-fidelity model is replaced by a computationally efficient approximation. Surrogate models, such as Gaussian processes and neural networks, can emulate the input-output relationship of the original model at a fraction of the computational cost. By strategically sampling the parameter space and constructing these surrogate models, engineers and researchers can explore uncertainty quantification, sensitivity analysis, and optimisation techniques that would otherwise be prohibitively expensive.

The complete code used to generate all results and figures in this post is available in the following repository: [model-calibration](https://github.com/mark-hobbs/articles/tree/main/model-calibration)

## Model

The `Model` class serves as a seamless interface between the numerical model and different *outer-loop* applications, such as design of experiments, optimisation and uncertainty quantification. The `Model` class simplifies the integration of numerical models into broader computational workflows.

When the `Model` is computationally expensive we must employ a surrogate model and the `SurrogateModel` class must also maintain the same common interface.

Pass an instance of a `Model` or `SurrogateModel` with a `__call__` method that computes a performance (fitness) metric, such as the mean squared error (MSE), for the given input parameters.

```python
class Model:
    """
    Base class for models used in optimisation and Bayesian inference.

    Subclasses must implement the run and __call__ method.
    """

    def __init__(self, x, y_observed=None):
        """
        x : list or array-like
            The independent variables (e.g., strain).

        y_observed : list or array-like
            The observed values of the dependent variables (e.g., stress).
        """
        self.x = x
        self.y_observed = y_observed

    def run(self):
        """
        Run the model to generate predictions

        This method must be implemented by subclasses and should
        return predictions computed by the model based on the 
        provided parameters.

        Returns
        -------
        list
            Model predictions corresponding to the input variables.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def __call__(self, *args, **kwargs):
        """
        Evaluate the performance (fitness) of the model
        
        Quantify how well the model predictions match 
        the observed data (e.g., mean squared error).

        Parameters
        ----------
        *args : tuple
            Positional arguments.

        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        float
            A score representing the performance of the model.
        """
        raise NotImplementedError("Subclasses must implement __call__ method.")
```

## Surrogate model 

The objective is to develop a generic surrogate model class for approximating the... forward model or likelihood function?

In Bayesian inference, a surrogate model like a Gaussian Process (GP) is often used to approximate the likelihood function when the true likelihood is computationally expensive or intractable. There are two common approaches:

### 1. **Likelihood Approximation**
   - The GP is used to learn a surrogate for the likelihood function $p(y \mid \theta)$.
   - Given a set of evaluated likelihoods $p(y \mid \theta_i)$ at training points $\theta_i$, the GP provides a probabilistic interpolation over the parameter space.
   - This is useful in settings like Approximate Bayesian Computation (ABC) where likelihoods are expensive to evaluate.

### 2. **Posterior Approximation (Bayesian Optimization for Inference)**
   - The GP models the log-posterior $\log p(\theta \mid y)$ instead of the likelihood directly.
   - This is often coupled with Bayesian Optimization (BO) to efficiently explore the posterior.
   - The acquisition function guides the selection of new points to evaluate (e.g., Expected Improvement or Upper Confidence Bound).

### Alternative: **Emulating the Forward Model**
   - Instead of approximating the likelihood directly, a GP can also be used to learn a surrogate of the forward model $f(\theta)$ where $y = f(\theta) + \epsilon$.
   - The GP then provides an implicit likelihood via its predictive distribution.

The surrogate model inherits from the real model...

```python
class Model:

  def __init__():
    pass

  def __call__():
    pass

  def evaluate():
    """
    Quantify the discrepancy between a given set of observed data and model predictions
    """
    pass

class SurrogateModel(Model):

  def __init__():
    pass
```

## Design of Experiments


## Likelihood

If the surrogate model is a Gaussian process, the model uncertainty can be incorporated into the observation noise.

## Calibration