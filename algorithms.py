import random, copy, numpy as np
from typing import Callable
from constraints import num_packages
from package import evaluate_solution, generate_random_solution

# switches 2 random packages in the final solution
def get_neighbor_solution(solution: list[int]) -> list[int]:
    neighbor_solution = copy.deepcopy(solution)
    i, j = random.sample(range(len(solution)), 2)
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution

def hill_climbing(neighbor_function: Callable [[list[int]], list[int]], iterations: int = 1000) -> list[int]:

    best_solution = random.sample(range(num_packages), num_packages)
    best_cost = evaluate_solution(best_solution)

    for _ in range(iterations):
        solution = neighbor_function(best_solution)
        cost = evaluate_solution(solution)

        if cost < best_cost:
            best_cost = cost
            best_solution = solution

    return best_solution, best_cost

# simulated annealing
def simulated_annealing(neighbor_function: Callable[[list[int]], list[int]], iterations: int = 1000, temperature: float = 1.0, cooling: float = 0.0) -> list[int]:
    solution = generate_random_solution()
    cost = evaluate_solution(solution)
    best_solution = solution
    best_cost = cost

    for _ in range(iterations):
        neighbor_solution = neighbor_function(solution)
        neighbor_cost = evaluate_solution(neighbor_solution)
        delta = cost - neighbor_cost
        acceptance_probability = np.exp(delta / temperature)
        
        if neighbor_cost < cost or random.random() < acceptance_probability:
            solution = neighbor_solution
            cost = neighbor_cost

        if cost < best_cost:
            best_solution = solution
            best_cost = cost

        temperature *= 1 - cooling
    return best_solution



# Genetic Algorithm 


# crossover functions ?? Devem retornar 2 soluções ou só uma?

# Partially mapped crossover
# This algorithm picks a random crossover point.
# It then exchanges the first part of each parent by swapping genes in the wrong place within each parent.
# This avoids duplicate genes and ensures that all genes are present in the offspring.  
def pmx_crossover(solution_1, solution_2):
    crossover_index  = np.random.randint(1, len(solution_1))
    child_1 = copy.deepcopy(solution_1)
    child_2 = copy.deepcopy(solution_2)
    for i in range(crossover_index):
        if child_1[i] != solution_2[i]:
            change_index = np.where(child_1 == solution_2[i])[0][0]
            child_1[i], child_1[change_index] = child_1[change_index], child_1[i]

        if child_2[i] != solution_1[i]:
            change_index = np.where(child_2 == solution_1[i])[0][0]
            child_2[i], child_2[change_index] = child_2[change_index], child_2[i]

    return child_1, child_2


# mutation functions

def swap_mutation(solution):
    index_1 = np.random.randint(0, len(solution))
    index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    solution[index_1], solution[index_2] = solution[index_2], solution[index_1]
    return solution


def inversion_mutation(solution : list[int])-> list[int]:
    index_1 = np.random.randint(0, len(solution))
    index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    solution[index_1:index_2] = solution[index_1:index_2][::-1]
    return solution

def scramble_mutation(solution: list[int])-> list[int]: 
    index_1 = np.random.randint(0, len(solution))
    index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    np.random.shuffle(solution[index_1:index_2])
    return solution

def random_mutation(solution : list[int]) -> list[int]:
    mutation = np.random.choice([swap_mutation, inversion_mutation, scramble_mutation])
    return mutation(solution)

# selection functions

#tournament solution

# returns the solution with highest scrore from a population
def greatest_fit(population: list[list[int]]) -> list[int]:  # podemos ter de retornar a score para além da solução
    best_solution = population[0]
    best_score = evaluate_solution(best_solution)
    for sol in population:
        score = evaluate_solution(sol)
        if score > best_score:
            best_solution = sol
            best_score = score
    return best_solution
    
# Returns a list of random solutions
def generate_population(population_size :int)-> list[int]:
    solutions = []
    for _ in range(population_size):
        solutions.append(generate_random_solution())
    return solutions

# Randomly chooses tournament_size solutions from the population and returns the one with the highest scores
def tournament_selection(population :list[list[int]], tournament_size:int = 20):
    if len(population) < tournament_size:
        return greatest_fit(population)
    participants = random.sample(population, tournament_size)
    return greatest_fit(participants)
    
# roulette wheel selection
def roulette_wheel_selection(population: list[list[int]]) -> list[int]:
    scores = [evaluate_solution(sol) for sol in population]
    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]
    return population[np.random.choice(len(population), p=probabilities)]


# Randomly chooses between tournament selection and roulette wheel selection
def random_selection(population : list[list[int]], tournament_size = 20) -> list[int]:
    selection_function = np.random.choice([tournament_selection, roulette_wheel_selection])
    if selection_function == tournament_selection:
        return selection_function(population, tournament_size)
    else:
        return selection_function(population)


# Perguntar ao stor como é que devemos fazer a seleção dos pais da próxima geração
# 2 childs por cada ou só uma?
# podemos ter de alterar o crossover para simplificar
# Também podemos adicionar o tamanho do torneio como argumento da função. Para já estamos a usar um default
def genetic_algorithm(crossover_function, population_size=100, generations=1000, mutation_rate=0.01):
    population = generate_population(population_size)
    best_solution = population[0]
    best_cost = evaluate_solution(best_solution)

    for _ in range(generations):
        new_population = []
        for solution in population:
            cost = evaluate_solution(solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
            
            parent1 = random_selection(population)
            parent2 = random_selection(population)
            child1, child2 = crossover_function(parent1, parent2)
            if np.random.rand() < mutation_rate:
                child1 = random_mutation(child1)
            # if np.random.rand() < mutation_rate:
            #     child2 = random_mutation(child2)
            new_population.append(child1)
            # new_population.append(child2)

        population = new_population

    return best_solution, best_cost