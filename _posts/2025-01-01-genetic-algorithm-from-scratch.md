---
layout: post
author: Mark Hobbs
title: Genetic algorithm from scratch
draft: True
---

View the repo: [ga-demo](https://github.com/mark-hobbs/ga-demo/)

This post details the implementation of a genetic algorithm from scratch using good object-oriented programming (OOP) practices and provides a number of visual examples to help develop the readers understanding.

Genetic algorithms are heuristic optimisation techniques based loosely on evolutionary theory, that mimic the process of natural selection to find solutions to complex optimisation problems.

In the natural world, evolution proceeds through reproduction, mutation and selection - where advantageous traits increase survival probability and are passed to future generations. This "survival of the fittest" mechanism drives biological adaptation over time. Genetic algorithms mimic this elegant natural process, creating evolving populations of solutions that progressively improve to solve complex optimisation challenges.

The core components of a genetic algorithm include:
- A population of potential solutions (individuals)
- A fitness function to evaluate how good each solution is
- Selection mechanisms to choose parents for reproduction
- Crossover and mutation operators to create new offspring
- Replacement strategies to form the next generation

![](/assets/images/ga-1.gif)
<figcaption>GA</figcaption>

## Implementation

By implementing a genetic algorithm with clean OOP design, we can create a flexible and reusable framework that separates concerns and makes the genetic algorithm adaptable to various optimisation problems without having to rewrite the core evolutionary logic.

Genetic algorithms provide an excellent demonstration of the principles and advantages of object-oriented programming. By designing classes that encapsulate key components — such as the evolutionary process, population and individuals — we can create well-defined abstractions that promote modularity, flexibility and a clear separation of concerns.

- `GeneticAlgorithm`: A class to control the evolutionary process.
- `Population`: The population represents a generation of individuals...
- `Individual`: Represents a candidate solution and encapsulates its genetic representation and fitness evaluation.
- Crossover method
- Mutation method

We provide a high-level overview of the implemented classes, focussing on their design and interactions. For the sake of simplicity, the code used to generate the animations and figures is omitted but can be found in the [repo](https://github.com/mark-hobbs/ga-demo/).

### `GeneticAlgorithm`

```python
class GeneticAlgorithm:

    def __init__(
        self,
        population,
        num_generations=50,
        num_parents=4,
        mutation_probability=0.05
    ):
        self.population = population
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.mutation_probability = mutation_probability
        self.fitness = []

    def generate_offspring(self):
        new_population = []
        for _ in range(len(self.population.individuals)):
            parent_a, parent_b = random.sample(self.population.parents, 2)
            child = parent_a.crossover(parent_b)
            child.mutate(self.mutation_probability)
            new_population.append(child)

        self.population.individuals = new_population

    def evolutionary_cycle(self):
        self.population.evaluate()
        self.fitness.append(max(self.population.fitness))
        self.population.select_parents(self.num_parents)
        self.generate_offspring()

    def evolve(self):
        for _ in tqdm(range(self.num_generations), desc="Evolution"):
            self.evolutionary_cycle()
```

### `Population`

```python
class Population:

    def __init__(self, individuals):
        self.individuals = individuals
        self.fitness = []
        self.parents = []

    def evaluate(self):
        self.fitness = [individual.fitness() for individual in self.individuals]

    def select_parents(self, num_parents):
        self.parents = sorted(
            self.individuals, key=lambda x: x.fitness(), reverse=True
        )[:num_parents]
```

### `Individual`

```python
class Individual:

    def __init__(self, genes):
        self.genes = genes
        self._fitness = None

    def evaluate_fitness(self):
        """
        This method should be implemented in subclasses to evaluate fitness.
        """
        raise NotImplementedError

    def fitness(self):
        """
        Returns the fitness of the individual. Computes fitness if it has not
        been computed yet.
        """
        if self._fitness is None:
            self.evaluate_fitness()
        return self._fitness

    def crossover(self, partner):
        """
        Crossover: combine the genetic material of two parent individuals 
        to create offspring

        Returns
        -------
        child
        """
        return self.crossover_method(self, partner)

    def mutate(self, mutation_probability):
        """
        Mutation: introduce small variations in the genetic material, 
        preventing premature convergence and maintaining diversity 
        in the population.
        """
        return self.mutate_method(self, mutation_probability)
```

## Problem

![](/assets/images/ga-2.gif)