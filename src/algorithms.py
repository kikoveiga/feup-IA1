import random, copy, numpy as np
from typing import Callable, Tuple
from package import Package, evaluate_solution, generate_random_solution, generate_package_stream

# switches 2 random packages in the solution
def get_neighbor_solution(solution: list[int]) -> list[int]:
    neighbor_solution: list[int] = copy.deepcopy(solution)
    i, j = random.sample(range(len(solution)), 2)
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution

def get_all_neighbors(solution: list[int]) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor: list[int] = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def hill_climbing(package_stream: list[Package], neighbor_function: Callable [[list[int]], list[int]], iterations: int = 1000) -> Tuple[list[int], float]:

    best_solution: list[int] = generate_random_solution()
    best_cost = evaluate_solution(package_stream, best_solution)

    for _ in range(iterations):
        solution: list[int] = neighbor_function(best_solution)
        cost = evaluate_solution(package_stream, solution)

        if cost < best_cost:
            best_cost = cost
            best_solution = solution

    return best_solution, best_cost

# simulated annealing
def simulated_annealing(package_stream: list[Package], neighbor_function: Callable[[list[int]], list[int]], iterations: int = 1000, temperature: float = 1.0, cooling: float = 0.0) -> Tuple[list[int], float]:
    solution: list[int] = generate_random_solution()
    cost = evaluate_solution(package_stream, solution)
    best_solution = solution
    best_cost = cost

    for _ in range(int(iterations)):

        if temperature <= 0:
            break
        
        neighbor_solution:list[int] = neighbor_function(solution)
        neighbor_cost = evaluate_solution(package_stream, neighbor_solution)
        delta = cost - neighbor_cost
        acceptance_probability = np.exp(delta / temperature)

        
        if neighbor_cost < cost or random.random() < acceptance_probability:
            solution = neighbor_solution
            cost = neighbor_cost

        if cost < best_cost:
            best_solution = solution
            best_cost = cost

        temperature *= 1 - cooling
    return best_solution, best_cost

def tabu_search(package_stream: list[Package], tabu_list_size: int = 100, iterations: int = 1000) -> list[int]:
    initial_solution: list[int] = generate_random_solution()
    best_solution: list[int] = initial_solution
    best_cost: float = evaluate_solution(package_stream, best_solution)
    current_solution: list[int] = initial_solution
    tabu_list: list[tuple[int, ...]] = []

    for _ in range(iterations):
        neighbors: list[list[int]] = get_all_neighbors(current_solution)
        neighbors_filtered: list[tuple[int, ...]] = [tuple(solution) for solution in neighbors if tuple(solution) not in tabu_list]

        if not neighbors:
            break

        next_solution: list[int] = min(neighbors, key=lambda solution: evaluate_solution(package_stream, solution))
        next_cost: float = evaluate_solution(package_stream, next_solution)

        if next_cost < best_cost:
            best_solution, best_cost = next_solution, next_cost

        tabu_list.append(tuple(next_solution))
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        
        current_solution = next_solution

    return best_solution


# Genetic Algorithm 


# crossover functions ?? Devem retornar 2 soluções ou só uma?

# Partially mapped crossover
# This algorithm picks a random crossover point.
# It then exchanges the first part of each parent by swapping genes in the wrong place within each parent.
# This avoids duplicate genes and ensures that all genes are present in the offspring.  
def pmx_crossover(solution_1: list[int], solution_2: list[int]) -> Tuple[list[int], list[int]]:
    crossover_index  = np.random.randint(1, len(solution_1))
    child_1 = copy.deepcopy(solution_1)
    child_2 = copy.deepcopy(solution_2)
    for i in range(crossover_index):
        if child_1[i] != solution_2[i]:
            change_indices = np.where(np.atleast_1d(child_1 == solution_2[i]))[0]
            if change_indices.size > 0:
                change_index = change_indices[0]
                child_1[i], child_1[change_index] = child_1[change_index], child_1[i]

        if child_2[i] != solution_1[i]:
            change_indices = np.where(np.atleast_1d(child_2 == solution_1[i]))[0]
            if change_indices.size > 0:
                change_index = change_indices[0]
                child_2[i], child_2[change_index] = child_2[change_index], child_2[i]

    return child_1, child_2


# mutation functions

def swap_mutation(solution: list[int]) -> list[int]:
    index_1: int = np.random.randint(0, len(solution))
    index_2: int = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    solution[index_1], solution[index_2] = solution[index_2], solution[index_1]
    return solution


def inversion_mutation(solution : list[int])-> list[int]:
    index_1: int = np.random.randint(0, len(solution))
    index_2: int = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    solution[index_1:index_2] = solution[index_1:index_2][::-1]
    return solution

def scramble_mutation(solution: list[int])-> list[int]: 
    index_1: int = np.random.randint(0, len(solution))
    index_2: int = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    np.random.shuffle(solution[index_1:index_2])
    return solution

def random_mutation(solution : list[int]) -> list[int]:
    mutation = random.choice([swap_mutation, inversion_mutation, scramble_mutation])
    return mutation(solution)

# selection functions

#tournament solution

# returns the solution with highest score from a population
def greatest_fit(package_stream: list[Package], population: list[list[int]]) -> list[int]:  # podemos ter de retornar a score para além da solução
    best_solution = population[0]
    best_score = evaluate_solution(package_stream, best_solution)
    for solution in population[1:]:
        score = evaluate_solution(package_stream, solution)
        if score > best_score:
            best_solution = solution
            best_score = score
    return best_solution
    
# Returns a list of random solutions
def generate_population(population_size: int)-> list[list[int]]:
    solutions = []
    for _ in range(population_size):
        solutions.append(generate_random_solution())
    return solutions

# Randomly chooses tournament_size solutions from the population and returns the one with the highest scores
def tournament_selection(package_stream: list[Package], population :list[list[int]], tournament_size:int = 20) -> list[int]:
    if len(population) < tournament_size:
        return greatest_fit(package_stream, population)
    participants = random.sample(population, tournament_size)
    return greatest_fit(package_stream, participants)
    
# roulette wheel selection
def roulette_wheel_selection(package_stream: list[Package], population: list[list[int]]) -> list[int]:
    scores = [evaluate_solution(package_stream, solution) for solution in population]
    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]
    return population[np.random.choice(len(population), p=probabilities)]


# Randomly chooses between tournament selection and roulette wheel selection
def random_selection(package_stream: list[Package], population : list[list[int]], tournament_size = 20) -> list[int]:
    selection_function: function = random.choice([tournament_selection, roulette_wheel_selection])
    if selection_function == tournament_selection:
        return tournament_selection(package_stream, population, tournament_size)
    else:
        return roulette_wheel_selection(package_stream, population)


# Perguntar ao stor como é que devemos fazer a seleção dos pais da próxima geração
# 2 childs por cada ou só uma?
# podemos ter de alterar o crossover para simplificar
# Também podemos adicionar o tamanho do torneio como argumento da função. Para já estamos a usar um default
def genetic_algorithm(package_stream: list[Package], crossover_function, population_size=100, generations=1000, mutation_rate=0.01):
    population = generate_population(population_size)
    best_solution = population[0]
    best_cost = evaluate_solution(package_stream, best_solution)

    for _ in range(generations):
        new_population = []
        for solution in population:
            cost = evaluate_solution(package_stream, solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
            
            parent1 = random_selection(package_stream, population)
            parent2 = random_selection(package_stream, population)
            child1, child2 = crossover_function(parent1, parent2)
            if np.random.rand() < mutation_rate:
                child1 = random_mutation(child1)
            # if np.random.rand() < mutation_rate:
            #     child2 = random_mutation(child2)
            new_population.append(child1)
            # new_population.append(child2)

        population = new_population

    return best_solution, best_cost

if __name__ == "__main__":
    
    # Test Hill Climbing
    #print("Hill Climbing:")
    #solution, cost = hill_climbing(generate_package_stream(), get_neighbor_solution, iterations=1000000)
    #print(f"Best solution: {solution} and cost: {cost}")

    # Test SA
    solution, cost = simulated_annealing(generate_package_stream(), get_neighbor_solution)
    print(f"Best solution: {solution}")

    #Test Genetic
    for _ in range(5):
        solution, cost = genetic_algorithm(generate_package_stream(), pmx_crossover, generations=1000, population_size=50)
        print(f"Best solution: {solution}")
        print(f"Best cost: {cost}")
