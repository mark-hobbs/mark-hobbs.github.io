---
layout: post
author: Mark Hobbs
title: Genetic algorithm from scratch
draft: True
---


This post details the implementation of a genetic algorithm from scratch and provides a number of visual examples to help develop the readers understanding. In addition to exploring the core concepts of genetic algorithms, this post places a strong emphasis on the adoption of good object-oriented programming (OOP) practices. Good code promotes clearer thinking, enabling the user to focus on the problem at hand.

Genetic algorithms are a class of heuristic optimisation techniques that mimic the process of natural selection. Beginning with a population of candidate solutions (or individuals), from which the best solutions (fittest individuals) are selected for reproduction (survival of the fittest). The fittest individuals produce offspring that form a new population (generation). The fittest individuals in each new generation are selected for breeding, and the process is repeated over many generations, leading to the emergence of increasingly optimal solutions.

All the code and examples are available in the following repo: [pyga](https://github.com/mark-hobbs/pyga/)

<!-- In the natural world, evolution proceeds through reproduction, mutation and selection - where advantageous traits increase survival probability and are passed to future generations. This "survival of the fittest" mechanism drives biological adaptation over time. Genetic algorithms mimic this elegant natural process, creating evolving populations of solutions that progressively improve to solve complex optimisation challenges. -->

## Implementation

By implementing a genetic algorithm with a clean object-oriented design, we can create a flexible and reusable framework that separates concerns and is adaptable to various optimisation problems without having to rewrite the core evolutionary logic.

A genetic algorithm can be broadly decomposed into the following logical units:

- `GeneticAlgorithm`: A class that controls the evolutionary process.
- `Population`: A class that represents a generation of individuals.
- `Individual`: A class that represents a candidate solution and encapsulates its genetic representation and fitness evaluation.
- `crossover(individual, partner)`: A function that combines the genes of two individuals to produce new offspring.
- `mutate(individual)`: A function that randomly mutates the gene of an individual in order to maintain diversity in the gene pool.

We provide a high-level overview of the implemented classes, focussing on their design and interactions. For the sake of simplicity, the code used to generate the animations and figures is omitted but can be found in the [repo](https://github.com/mark-hobbs/pyga/).

### `GeneticAlgorithm`

The `GeneticAlgorithm` class controls the evolutionary process. The user can select the number of generations and number of parents used for generating offspring. There is only a single public method... `evolve`

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

    def _generate_offspring(self):
        """
        Generate a new population by combining genetic material 
        from selected parents (high-fitness) and applying random 
        mutations to their offspring
        """
        new_population = []
        for _ in range(len(self.population.individuals)):
            parent_a, parent_b = random.sample(self.population.parents, 2)
            child = parent_a.crossover(parent_b)
            child.mutate(self.mutation_probability)
            new_population.append(child)

        self.population.individuals = new_population

    def _evolutionary_cycle(self):
        """
        Executes one iteration of the evolutionary process:
        - Evaluate the fitness of the population
        - Select parents for reproduction
        - Generates offspring to form the next generation
        """
        self.population.evaluate()
        self.fitness.append(max(self.population.fitness))
        self.population.select_parents(self.num_parents)
        self._generate_offspring()

    def evolve(self):
        """
        Runs the genetic algorithm for the specified number 
        of generations.
        """
        for _ in tqdm(range(self.num_generations), desc="Evolution"):
            self._evolutionary_cycle()
```

### `Population`

The `Population` class represents all individuals within a single generation. Methods are provided for evaluating the fitness of all individuals and selecting parents for reproduction.

Evaluating the fitness of each individual in the population is an embarrassingly parallel process. In the below implementation, the `evaluate` method computes the fitness of every individual in serial; however, this process could be easily parallelised on a traditional cluster or cloud infrastructure... Dask, Ray, Kubernetes (horizontal scaling)...

```python
class Population:

    def __init__(self, individuals):
        self.individuals = individuals
        self.fitness = []
        self.parents = []

    def evaluate(self):
        """
        Evaluate the fitness of every individual in the population
        """
        self.fitness = [individual.fitness() for individual in self.individuals]

    def select_parents(self, num_parents):
        """
        Rank individuals by fitness and select the strongest for reproduction
        """
        self.parents = sorted(
            self.individuals, key=lambda x: x.fitness(), reverse=True
        )[:num_parents]
```

### `Individual`

The `Individual` class represents a candidate solution and encapsulates its genetic representation and fitness evaluation. It serves as a base class, allowing subclasses to define problem-specific fitness functions. By inheriting from `Individual`, custom implementations can be created for various optimisation problems, making the package adaptable to different domains.

```python
class Individual:

    _crossover_method = None
    _mutation_method = None

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
        if self._crossover_method is None:
            raise NotImplementedError("Crossover method not defined.")
        return self._crossover_method(self, partner)

    def mutate(self, mutation_probability):
        """
        Mutation: introduce small variations in the genetic material, 
        preventing premature convergence and maintaining diversity 
        in the population.
        """
        if self._mutate_method is None:
            raise NotImplementedError("Mutation method not defined.")
        return self._mutate_method(self, mutation_probability)
```

### Crossover and mutation

To enable users to change the crossover and mutation methods in a general way, we will adopt a strategy design pattern. The chosen crossover and mutation strategy is dependent on the problem type (e.g. continuous, binary, permutation-based), allowing flexibility and extensibility without modifying the core genetic algorithm logic.

The function signatures for the crossover and mutation methods must follow a standard interface to ensure compatibility:

- `crossover(individual, partner)`: Takes two individuals and returns one or more offspring.
- `mutate(individual)`: Applies a mutation operation to a single individual and returns the mutated individual.

This design allows new strategies to be added as modular components, making the code adaptable to a wide range of optimisation problems.

| **Crossover Method**           | **Description** | **Problem Type** |
|--------------------------------|----------------|------------------|
| **One-Point Crossover**       | Single crossover point; genes swapped after this point. | Binary, Continuous |
| **Two-Point Crossover**       | Two crossover points; middle segment is swapped. | Binary, Continuous |
| **Uniform Crossover**         | Each gene is randomly taken from either parent. | Binary, Continuous |
| **Arithmetic Crossover**      | Offspring genes are a weighted sum of parent genes. | Continuous |
| **Blend Crossover (BLX-α)**   | Offspring genes are chosen from an extended range of parent values. | Continuous |
| **Simulated Binary Crossover (SBX)** | Mimics single-point crossover but for real-valued variables using probability distribution. | Continuous |
| **Partially Mapped Crossover (PMX)** | Ensures offspring inherit ordered subsets while avoiding duplicates. | Permutation (e.g., TSP) |
| **Order Crossover (OX)**      | Preserves sequence order while maintaining feasibility. | Permutation (e.g., TSP) |
| **Cycle Crossover (CX)**      | Ensures each gene comes from exactly one parent while preserving positional info. | Permutation (e.g., TSP) |
| **Heuristic Crossover**       | Offspring are a biased weighted combination of parents, favoring the fitter one. | Continuous |


| **Mutation Method**            | **Description** | **Problem Type** |
|--------------------------------|----------------|------------------|
| **Bit Flip Mutation**          | Randomly flips a bit (for binary representation). | Binary |
| **Swap Mutation**              | Two randomly chosen genes swap positions. | Permutation |
| **Scramble Mutation**          | A subset of genes is shuffled randomly. | Permutation |
| **Inversion Mutation**         | A segment of genes is reversed in order. | Permutation |
| **Gaussian Mutation**          | Real-valued genes are perturbed by adding Gaussian noise. | Continuous |
| **Uniform Mutation**           | A gene is replaced with a new random value within its range. | Continuous |
| **Polynomial Mutation**        | Perturbs genes using a polynomial probability distribution. | Continuous |
| **Non-uniform Mutation**       | Mutation step size decreases over generations to fine-tune solutions. | Continuous |
| **Boundary Mutation**          | A gene is set to either its minimum or maximum allowed value. | Continuous |
| **Adaptive Mutation**          | Mutation rate adjusts dynamically based on evolution progress. | Continuous |

## Packaging

By packaging the code, it can be easily shared, installed and used by others. Many engineers and researchers conflate general-purpose components such as solvers or optimisers with specific applications, resulting in tightly coupled code that is difficult to reuse or extend.

Distributing reusable code...

By separating reusable functionality from application specific logic and packaging...

... tightly coupled solutions into modular building blocks that can be installed, shared and maintained independently. 

When core components (like optimisation methods) are properly abstracted and packaged, they become valuable assets across multiple projects rather than remaining trapped in single-use implementations. This separation enhances code quality, promotes reusability and enables collaborative development.

The `pyga` package can be installed locally using the following command `pip install git+https://github.com/mark-hobbs/pyga.git`.

## Problems

We address two classes of optimisation problem; (1) a permutation based problem, and (2) a continuous problem. 

The clean abstractions afforded by good object-oriented design enable us to address different optimisation problems with the same package. The user must subclass `Individual`, implementing the gene representation, the fitness function and problem specific constraints... 

This approach ensures that the core optimisation logic remains general-purpose, making it easy to address different problems without modifying the underlying framework.

### Permutation type problem

```python
n_points = 10
points = [Point(np.random.rand(), np.random.rand()) for _ in range(n_points)]

population_size = 25
individuals = [Polygon(np.random.permutation(points)) for _ in range(population_size)]
population = Population(individuals)

ga = GeneticAlgorithm(
    population=population,
    num_generations=100,
    num_parents=4,
    mutation_probability=0.05
    )
ga.evolve()
```

![](/assets/images/ga-1.gif)

### Continuous type problem

![](/assets/images/ga-2.gif)

![](/assets/images/ga-3.gif)


## Conclusions

Genetic algorithms provide an excellent demonstration of the principles and advantages of object-oriented programming. By designing classes that encapsulate key components — such as the evolutionary process, population and individuals — we can create well-defined abstractions that promote modularity, flexibility and a clear separation of concerns.

The clean object-oriented design enables the user to efficiently generate an initial population, configure the genetic algorithm and initiate the evolutionary process with minimal code. Concise and readable code enables the user to focus on the actual problem...

Problem decomposition is fundamental to... Effective problem decomposition requires a deep understanding of the problem domain. Without a proper grasp of the core elements/concepts... creating inappropriate abstractions or inefficient divisions of functionality.
