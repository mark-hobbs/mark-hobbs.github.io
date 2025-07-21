---
layout: post
author: Mark Hobbs
title: Genetic algorithm from scratch
draft: True
---


This post details the implementation of a genetic algorithm from scratch and provides a number of visual examples to help develop the readers understanding. In addition to exploring the core concepts of genetic algorithms, this post places a strong emphasis on the importance of good object-oriented programming (OOP) practices and effective abstraction. Carefully designed abstractions are the key to writing usable and extensible software.

Genetic algorithms are a class of heuristic optimisation techniques that mimic the process of natural selection. Beginning with a population of candidate solutions (or individuals), from which the best solutions (fittest individuals) are selected for reproduction (survival of the fittest). The fittest individuals produce offspring that form a new population (generation). The fittest individuals in each new generation are selected for breeding, and the process is repeated over many generations, leading to the emergence of increasingly optimal solutions.

All the code and examples are available in the following repo: [pyga](https://github.com/mark-hobbs/pyga/)

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

The `GeneticAlgorithm` class controls the evolutionary process. The user must provide an initial population from which the evolutionary process starts. The user can also define the number of generations `num_generations` and number of parents `num_parents` used for generating offspring. The optimisation process is initiated by calling `evolve`.

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

Evaluating the fitness of each individual in the population is an embarrassingly parallel process. In the below implementation, the `evaluate` method computes the fitness of every individual in serial; however, this process could be easily parallelised on a traditional cluster or cloud infrastructure.

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

The `Individual` class represents a candidate solution and encapsulates its genetic representation and fitness evaluation. It serves as a base class, allowing subclasses to define problem-specific fitness functions. By inheriting from `Individual`, custom implementations can be created for various optimisation problems, making the package adaptable to different domains. Users must define the `_crossover_method` and `_mutation_method` when subclassing `Individual`. This is discussed further in the following section.

The `Individual` class essentially functions as an interface to the optimiser. The clear separation of concerns between the `Individual` and the optimisation routine enables the user to address any problem for which a genetic representation and fitness measure can be defined with ease. 

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

An obvious extension to the above would be to write a class - `HTTPIndividual(Individual)` - that exposes the `fitness` evaluation as an API endpoint. By decoupling the fitness evaluation from the local runtime, it becomes possible to parallelise the fitness evaluation of all individuals in the population on cloud-based infrastructure.

### Crossover and mutation

The chosen crossover and mutation strategy is dependent on the problem type (e.g. continuous, binary, permutation-based). To enable users to change the crossover and mutation methods without modifying the core genetic algorithm logic, we will adopt a strategy design pattern. 

The function signatures for the crossover and mutation methods must follow a standard interface to ensure compatibility:

- `crossover(individual, partner)`: Takes two individuals and returns one or more offspring.
- `mutate(individual)`: Applies a mutation operation to a single individual and returns the mutated individual.

Users define the `_crossover_method` and `_mutation_method` when subclassing `Individual`. This design allows new strategies to be added as modular components, making the code adaptable to a wide range of optimisation problems.

## Packaging

Packaging is an essential but often neglected step in the software development process... generally serves one of two distinct purposes: (1) solving a specific problem, or (2) building a general-purpose tool. The first addresses immediate needs and reflects the particulars of a single task. By contrast, the second involves recognising patterns that recur across problems and abstracting them into reusable components. For example, (1) might be the optimisation of a wing to maximise aerodynamic efficiency, while (2) would be the optimiser. 

Domain experts (e.g. engineers, researchers, data scientists) approach code as a sequence of steps (i.e. a script) that addresses a specific problem and rarely think about code as a reusable building block.

This approach produces problem-specific solutions that are hard to maintain and extend. Inexperienced programmers often couple general-purpose components (such as optimisers) and problem-specific logic, resulting in tightly coupled code that is difficult to reuse or extend.

... tightly coupled solutions into modular building blocks that can be installed, shared and maintained independently.

By separating reusable functionality from problem-specific logic, general-purpose components can be packaged and distributed. This approach reduces duplicate efforts and enables users to collaborate and build on prior work.

When core components (like optimisation methods) are properly abstracted and packaged, they become valuable assets across multiple projects rather than remaining trapped in single-use implementations. This separation enhances code quality, promotes reusability and enables collaborative development. By adopting this approach, domain experts can move from writing isolated scripts to building robust software libraries.

The `pyga` package can be installed locally using the following command `pip install git+https://github.com/mark-hobbs/pyga.git`.


## Problems

We address two classes of optimisation problem; (1) a permutation based problem, and (2) a continuous problem. 

The clean abstractions afforded by good object-oriented design enable us to address different optimisation problems with the same package. The user must subclass `Individual`, implementing the gene representation, the fitness function and problem specific constraints... 

This approach ensures that the core optimisation logic remains general-purpose, making it easy to address different problems without modifying the underlying framework.

### Permutation type problem

**Problem statement:** Given a set of points in 2D space, determine the polygon that maximises the ratio of its area to the square of its perimeter $(\text{Area}/\text{Perimeter}^2)$. This ratio serves as a measure of compactness, which is often desirable in various fields such as materials science, biology, and urban planning. Compact shapes can lead to more efficient designs, reduced material usage, and optimised spatial arrangements.

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

**Problem statement:** Given an initial set of points in 2D space, evolve their positions such that the resulting configuration closely matches a predefined target shape (e.g. a star). The fitness function quantitatively evaluates the similarity between a candidate shape and the target, guiding the evolution toward the optimal arrangement.

![](/assets/images/ga-2.gif)

![](/assets/images/ga-3.gif)

## Abstraction

We will briefly discuss the idea of abstraction and examine how it informs the design of the genetic algorithm package. Abstraction is a fundamental concept in software design, yet it is hard to precisely define.

Carefully designed abstractions are the key to writing usable and extensible code. Abstraction is the process of managing complexity by only exposing the essential features of an object/system and hiding the lower-level implementation details.

It is a common misconception that the primary purpose of abstraction is to remove **repetition**. Whilst it might to a degree serve this purpose, the most important job of a good abstraction is to remove dependencies. Good abstractions provide interfaces that reduce **coupling**...

Coupling is a measure of the degree of dependency between software modules. Stated another way, coupling is a measure of the amount of work required to modify a module. Strong coupling implies that changes in one module are likely to necessitate extensive modifications in connected modules, leading to a cascading effect.

A good design aims for low coupling, meaning modules are relatively independent and changes in one module have a minimal impact on others. 

The `Individual` class effectively demonstrates the principle of abstraction. ...by providing a consistent interface between the problem domain and the optimiser, enabling the same optimisation logic to be applied across a wide range of problems with minimal changes.

The `Population` class could be modified to evaluate individuals in parallel and no changes would be required to the `Individual` class...

- Single responsibility principle
- Separation of concerns
- Dependency extraction
- Loose-coupling
- Interfaces

[Essence of abstraction](https://reasonunderpressure.com/blog/posts/the-essence-of-abstraction)

## Conclusions

Genetic algorithms provide an excellent demonstration of the principles and advantages of object-oriented programming. By designing classes that encapsulate key components - such as the evolutionary process, population and individuals - we can create well-defined abstractions that promote modularity, flexibility and a clear separation of concerns.

The clean object-oriented design enables the user to efficiently generate an initial population, configure the genetic algorithm and initiate the evolutionary process with minimal code. Concise and readable code enables the user to focus on the actual problem...

Problem decomposition is fundamental to... Effective problem decomposition requires a deep understanding of the problem domain. Without a proper grasp of the core elements/concepts... creating inappropriate abstractions or inefficient divisions of functionality.

Perhaps the most significant difference between new and experienced programmers is their ability to identify powerful abstractions. Choosing the right abstractions requires both programming experience and domain specific knowledge. It is unlikely that an experienced programmer with no domain knowledge of optimisation would chose an appropriate abstraction.