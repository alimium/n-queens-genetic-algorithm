from copy import copy
import random
import numpy as np
random.seed(0)

class SelectionMethod:
    FITTEST = 'fittest'
    RANK = 'rank'
    TOURNAMENT = 'tournament'
    
class GenEvolve:
    def __init__(self, n, pop_size, max_generations):
        self.n = n
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.population = self._initialize_population()
        self.previous_population = self.population
        self.g=0

    def _initialize_population(self):
        # this ensures row conflicts will not happen
        return [random.choices(range(self.n),k=self.n) for i in range(self.pop_size)]
        # population = []
        # for i in range(self.pop_size):
        #     chromosome = list(range(self.n))
        #     random.shuffle(chromosome)
        #     population.append(chromosome)
        # return population


    def _loss(self, chromosome):
        # # number of conflicts
        # col_conflicts = 0
        # diag_conflicts = 0

        # # column conflicts
        # values,counts = np.unique(chromosome, return_counts=True)
        # col_conflicts = sum(c-1 for c in counts) 

        # # diagonal conflicts
        # for i in range(self.n):
        #     for j in range(i+1, self.n):
        #         if abs(i-j) == abs(chromosome[i]-chromosome[j]):
        #             diag_conflicts += 1

        # # diagonal and column conflicts have the same weight
        # return col_conflicts + diag_conflicts
    
        # number of conflicts
        col_conflicts = 0
        diag_conflicts = 0

        # sets to keep track of conflicts
        col_set = set()
        diag_set1 = set()
        diag_set2 = set()

        for i in range(self.n):
            col = chromosome[i]
            diag1 = i - col
            diag2 = i + col

            # check for column conflicts
            if col in col_set:
                col_conflicts += 1
            else:
                col_set.add(col)

            # check for diagonal conflicts
            if diag1 in diag_set1:
                diag_conflicts += 1
            else:
                diag_set1.add(diag1)

            if diag2 in diag_set2:
                diag_conflicts += 1
            else:
                diag_set2.add(diag2)

        # diagonal and column conflicts have the same weight
        return col_conflicts + diag_conflicts

    def _selection(self, population):
        # Select the fittest chromosomes for reproduction
        sorted_candidates = sorted(population, key=self._loss)
        selected_population = sorted_candidates[:self.pop_size]
        return selected_population

    def _rank_based_selection(self, population):
        # randomly select a chromosome based on its loss
        fitness_values = [self._loss(chromosome) for chromosome in population]
        ranks = list(range(1, self.pop_size+1))
        sum_rank = sum(ranks)
        rank_probabilities = [rank / sum_rank for rank in ranks]
        selected_population = random.choices(population, weights=rank_probabilities, k=self.pop_size)
        return selected_population

    def _tournament_selection(self,population, k=2):
        parents = []
        for i in range(self.pop_size):
            competitors = random.sample(population, k)
            winner = min(competitors, key=self._loss)
            parents.append(winner)
        return parents

    def _crossover(self, parent1, parent2): 
        # Select a random crossover point (single point crossover)
        crossover_point = random.randint(0, self.n - 1)

        # Create the offspring chromosomes by swapping the genetic material of the parents
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2
    
    def _mutation(self, chromosome):
        # Introduce random changes in the chromosome to maintain genetic diversity
        # The mutation function should return the mutated chromosome
        if random.random() < 0.5:
            # Select two random positions in the chromosome and swap their values
            i, j = random.sample(range(self.n), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def evolve(self, selection=SelectionMethod.TOURNAMENT, full_restart=True):
        best_fitness = np.inf
        stagnation_count = 0
        max_stagnation = self.n*3

        while self.g < self.max_generations:

            if selection == SelectionMethod.RANK:
                parents = self._rank_based_selection(self.population)
            elif selection == SelectionMethod.TOURNAMENT:
                parents = self._tournament_selection(self.population)
            else:
                parents = self._selection(self.population)

            new_generation = []
            while len(new_generation)<self.pop_size:
                # choose parents
                parent1, parent2 = random.sample(parents, 2)

                # crossover
                child1, child2 = self._crossover(parent1, parent2)
                # mutation
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                family = []
                family.append(parent1)
                family.append(parent2)
                family.append(child1)
                family.append(child2)
                family = sorted(family, key=self._loss)
                new_generation.append(family[0])
                new_generation.append(family[1])

            best_chromosome = min(new_generation, key=self._loss)
            best_chromosome_fitness = self._loss(best_chromosome)
            if best_chromosome_fitness == 0:
                return best_chromosome
            elif best_chromosome_fitness < best_fitness:
                best_fitness = best_chromosome_fitness
                stagnation_count = 0
                self.previous_population = copy(self.population)
            else:
                stagnation_count += 1

            self.population = new_generation
            self.g += 1

            # check for stagnation
            if stagnation_count >= max_stagnation:
                print(f"Generation {self.g-1:<7}The algorithm may have reached a plateau. Restarting...")
                if full_restart:
                    self.population = self._initialize_population()
                else:
                    self.population = self.previous_population
                best_fitness = np.inf
                stagnation_count = 0
            else:
                print(f'Generation {self.g-1:<7}Fittest: {best_chromosome_fitness:<4}Global Fittest: {best_fitness}')
        print("Maximum generations reached. No solution found.")
        return None


    def print_board(self, queens):  # CODE GENERATED BY BING AI
        print(f'\nSolution found on generation {self.g}')
        # queens is a list of length n where each element is a number between 0 and n-1
        # that shows the index of the cell where the queen is located in that row
        n = len(queens)  # get the size of the board
        border = "+" + "+".join(["-" * 4] * n) + "+"  # create a border string
        print(border)  # print the border string
        for i in range(n):  # loop over the rows
            row = "|"  # initialize a string for the row with a |
            for j in range(n):  # loop over the columns
                if queens[i] == j:  # if there is a queen at this position
                    row += " Q  |"  # add a Q and two | to the row string
                else:  # otherwise
                    row += "    |"  # add a . and two | to the row string
            print(row)  # print the row string
            print(border)  # print the border string

