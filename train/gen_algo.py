
# import numpy as np
import os
from tqdm import tqdm
import torch
import numpy as np

os.system('clear')

def print_array(a):
    [print(round(c, 2)) for c in a]

class GeneticAlgorithm:
    def __init__(self,
            fitness_function=None,
            random_genome_value=None,
            maximize=True,
            mutation=.1,
            elitism=.4,
            maxPopulationSize=10
        ) -> None:
        self.mutation = mutation
        self.elitism = elitism
        self.maxPopulationSize = maxPopulationSize
        self.fitness_function = fitness_function
        self.random_genome_value = random_genome_value
        self.maximize_fitness = maximize
    
    def init_population(self, chromosomes):
        n = len(chromosomes)
        self.population = []

        # Generate the population by performing crossover between random pairs of chromosomes
        while len(self.population) < self.maxPopulationSize:
            # Randomly select two different chromosomes from the list
            c1, c2 = torch.randperm(n)[:2]  # Get two random indices
            c1 = chromosomes[c1]
            c2 = chromosomes[c2]

            # Perform crossover between the two selected chromosomes
            children = self.crossover_iterate(c1=c1, c2=c2, difference=0.5, n_of_children=3)

            remaining_slots = self.maxPopulationSize - len(self.population)

            # Add only the number of children that can fit into the population without exceeding the limit
            if len(children) > remaining_slots:
                children = children[:remaining_slots]
            
            self.population.extend(children)

    def evolve(self, crossover_param=.5, generations=500):
        graph = []
        # assert current population
        for gen in tqdm(range(generations)):
            metrics = []
            for g in self.population:
                fitness = self.fitness_function(g)
                metrics.append({'gene': g, 'fitness': fitness})

            metrics_sorted = sorted(metrics, key=lambda x: x['fitness'], reverse=self.maximize_fitness)

            # cut population by half
            metrics_sorted = metrics_sorted[:int(len(metrics_sorted) / 2)]
            
            # apply elitism
            n_the_elite = int(self.elitism * len(metrics_sorted))
            the_elite = metrics_sorted[:n_the_elite]

            graph.append([gen, the_elite[0]['fitness']])

            common_folk = metrics_sorted[n_the_elite:]

            new_population = [c['gene'] for c in the_elite]

            while len(new_population) < self.maxPopulationSize:
                    
                c1 = np.random.choice(the_elite)
                c2 = np.random.choice(common_folk)

                children = self.crossover_iterate(
                    c1=c1['gene'],
                    c2=c2['gene'],
                    difference=crossover_param,
                    n_of_children=3,
                    mutation=self.mutation
                )

                new_population += children
            
            if len(new_population) > self.maxPopulationSize:
                new_population = new_population[:self.maxPopulationSize]

            self.population = new_population
                
        return graph

    def get_best(self):
        return self.population[0]
    
    def crossover_iterate(self, c1, c2, difference=0.5, mutation=0.0, n_of_children=3):
        if len(c1) != len(c2):
            raise Exception("Chromosome lengths don't match")
        
        children = []
        len_c = len(c1)
        num_samples = int(difference * len_c)

        # Generate mutation masks and random mutation values

        for _ in range(n_of_children):
            # Clone the first chromosome
            child = c1.clone()

            # Crossover: Select random indices to replace with values from c2
            i_to_change = torch.randperm(len_c)[:num_samples]
            child[i_to_change] = c2[i_to_change]

            # Mutation: Apply mutation using the precomputed mutation mask and values
            mutation_mask = torch.rand(len_c) < mutation  # Boolean mask for mutation
            mutation_values = self.random_genome_value(len_c)  # Generate batch of mutation values
            child[mutation_mask] += mutation_values[mutation_mask]

            children.append(child)
        
        return children

    
    def crossover_middle(self, c1, c2, padding=.2, n_of_children=3):
        if len(c1) != len(c2):
            raise Exception("Chromosome lengths dont match")
        children = []
        for _ in range(n_of_children):
            crossover_percent = .5 + torch.choice.uniform(-padding, padding)
            crossover_point = int(crossover_percent * len(c1))
            child = c1[:crossover_point] + c2[crossover_point:]
            children.append(child)
        return children


    