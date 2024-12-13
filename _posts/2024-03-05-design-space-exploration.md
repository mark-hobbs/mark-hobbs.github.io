---
layout: post
author: Mark Hobbs
title: Design space exploration
---

This post provides a visual guide to design space exploration and its connection to optimisation and uncertainty quantification.

Explore the repo: [design-space-exploration](https://github.com/mark-hobbs/design-space-exploration)

**Contents**
- [1. Motivation](#1-motivation)
- [2. Design space](#2-design-space)
- [3. Grid search](#3-grid-search)
- [4. Monte Carlo (MC)](#4-monte-carlo-mc)
- [5. Markov Chain Monte Carlo (MCMC)](#5-markov-chain-monte-carlo-mcmc)
- [6. Optimisation](#6-optimisation)
- [7. Related concepts](#7-related-concepts)

## 1. Motivation

Efforts at the intersection of machine learning and simulation generally fall into two main categories: (1) accelerating simulations and (2) enhancing design space exploration and optimisation. These categories are interconnected, forming a feedback loop where advances in one area can significantly impact the other. Accelerating simulations can unlock *outer-loop* applications that are currently infeasible due to the high computational cost associated with numerous repeated simulations. Conversely, improving design space exploration can drastically reduce the number of simulations needed to identify optimal designs or strategically navigate the design space to minimise uncertainty. This post provides a visual guide to design space exploration, aiming to build intuition without relying on formal mathematics.

## 2. Design space

Contour plot illustrating the relationship between two design parameters, $X_1$ and $X_2$, and the corresponding values of the objective function. In many engineering applications, evaluating this objective function can be extremely computationally expensive.

![](/assets/images/design-space.png)

## 3. Grid search

Each point represents a single run of a computationally expensive simulation

![](/assets/images/grid-search.png)

## 4. Monte Carlo (MC)

![](/assets/images/monte-carlo.png)

## 5. Markov Chain Monte Carlo (MCMC)

MCMC samplers systematically sample the design space in a way that favours regions with higher probability or likelihood, effectively focusing the exploration on the most probable designs. This approach is particularly useful when the goal is to understand the distribution of optimal solutions or identify regions of the design space that meet specific criteria, rather than exhaustively searching all possibilities.

![](/assets/images/mcmc.png)
![](/assets/images/mcmc-animation.gif)

## 6. Optimisation

The primary goal of optimisation is to find the best solution(s) according to specific objective criteria. The process inherently involves exploring the design space to identify regions that offer optimal performance. Techniques such as gradient-based optimisation, genetic algorithms, or Bayesian optimisation navigate the design space by sampling points, evaluating their performance, and iteratively refining the search. Thus, optimisation not only seeks the best designs but also contributes to understanding the structure and characteristics of the design space.

**Gradient-based optimisation**

![](/assets/images/optimisation.png)

**Genetic algorithm**

![](/assets/images/genetic-algorithm.gif)

## 7. Related concepts

Generative models aim to infer the underlying distribution from which observed data is generated.
