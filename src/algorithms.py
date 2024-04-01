import random, copy, numpy as np
from typing import Callable
from package import Package, evaluate_solution, generate_random_solution, generate_package_stream
from queue import PriorityQueue

test_packages: list[Package] = [Package('urgent', (56.97556558957435, 27.90045109824626)),
                                Package('fragile', (44.8203921010497, 45.239271577485084)),
                                Package('normal', (3.23756109561091, 21.1712638842833)),
                                Package('normal', (31.342549842605905, 43.79155490345392)),
                                Package('normal', (54.51149696067019, 55.917516798837994)),
                                Package('normal', (0.13982076073304261, 14.455164572152892)),
                                Package('fragile', (59.311394005015615, 52.17887506703184)),
                                Package('fragile', (43.07637979465492, 7.6826233225053775)),
                                Package('fragile', (43.662203998731016, 33.37818643749602)),
                                Package('urgent', (23.530958559725182, 3.3941997983349426))]

test_packages = generate_package_stream()

# switches 2 random packages in the solution
def get_random_neighbor(package_stream: list[Package], solution: list[int]) -> tuple[list[int], float]:
    neighbor_solution: list[int] = copy.deepcopy(solution)
    i, j = random.sample(range(len(solution)), 2)
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution, evaluate_solution(package_stream, neighbor_solution)

# switches 2 packages in an ordered way until a better solution is found
def get_first_better_neighbor(package_stream: list[Package], solution: list[int]) -> tuple[list[int], float]:
    cost: float = evaluate_solution(package_stream, solution)
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor_solution: list[int] = copy.deepcopy(solution)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbor_cost = evaluate_solution(package_stream, neighbor_solution)
            if neighbor_cost < cost:
                return neighbor_solution, neighbor_cost
    return solution, cost

# switches 2 packages in an ordered way until the best solution is found
def get_best_neighbor(package_stream: list[Package], solution: list[int]) -> tuple[list[int], float]:
    best_solution: list[int] = solution
    best_cost: float = evaluate_solution(package_stream, solution)
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor_solution: list[int] = copy.deepcopy(solution)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbor_cost = evaluate_solution(package_stream, neighbor_solution)
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_solution = neighbor_solution
    return best_solution, best_cost
    
def get_all_neighbors(package_stream: list[Package], solution: list[int]) -> PriorityQueue:
    neighbors: PriorityQueue = PriorityQueue()
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor_solution: list[int] = copy.deepcopy(solution)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbors.put((evaluate_solution(package_stream, neighbor_solution), neighbor_solution))
    return neighbors

def hill_climbing(package_stream: list[Package], neighbor_function: Callable [[list[Package], list[int]], tuple[list[int], float]], iterations: int = 1000) -> tuple[list[int], float]:

    solution: list[int] = generate_random_solution()
    cost: float = evaluate_solution(package_stream, solution)
    best_solution: list[int] = solution
    best_cost: float = cost

    for _ in range(iterations):
        solution, cost = neighbor_function(package_stream, best_solution)
        
        if cost < best_cost:
            best_cost = cost
            best_solution = solution

        elif neighbor_function != get_random_neighbor:
            solution = generate_random_solution()
            cost = evaluate_solution(package_stream, solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

    return best_solution, best_cost

# simulated annealing
def simulated_annealing(package_stream: list[Package], iterations: int = 1000, temperature: float = 1.0, cooling: float = 0.01) -> tuple[list[int], float]:
    solution: list[int] = generate_random_solution()
    cost: float = evaluate_solution(package_stream, solution)
    best_solution: list[int] = solution
    best_cost: float = cost

    for _ in range(iterations):

        if temperature <= 0:
            print(iterations)
            break
        
        neighbor_solution, neighbor_cost = get_random_neighbor(package_stream, solution)
        delta: float = neighbor_cost - cost

        if delta / temperature > 700:
            acceptance_probability = float('inf')
        else:
            acceptance_probability = np.exp(delta / temperature)

        
        if delta < 0 or random.random() < acceptance_probability:
            solution = neighbor_solution
            cost = neighbor_cost

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        temperature *= 1 - cooling

    return best_solution, best_cost

def tabu_search(package_stream: list[Package], tabu_list_size: int = 1000, iterations: int = 1000) -> tuple[list[int], float]:
    
    best_solution: list[int] = generate_random_solution()
    best_cost: float = evaluate_solution(package_stream, best_solution)
    current_solution: list[int] = best_solution
    tabu_solutions: list[tuple[int, ...]] = []

    for _ in range(iterations):
        neighbors: PriorityQueue = get_all_neighbors(package_stream, current_solution)

        next_cost, next_solution = neighbors.get()

        if tuple(next_solution) in tabu_solutions:
            while tuple(next_solution) in tabu_solutions and not neighbors.empty():
                next_cost, next_solution = neighbors.get()
            if neighbors.empty():
                break

        if next_cost < best_cost:
            best_solution, best_cost = next_solution, next_cost

        tabu_solutions.append(tuple(next_solution))
        if len(tabu_solutions) > tabu_list_size:
            tabu_solutions.pop(0)
        
        current_solution = next_solution

    return best_solution, best_cost

# Partially mapped crossover
# This algorithm picks a random crossover point.
# It then exchanges the first part of each parent by swapping genes in the wrong place within each parent.
# This avoids duplicate genes and ensures that all genes are present in the offspring.  
def pmx_crossover(solution_1: list[int], solution_2: list[int]) -> tuple[list[int], list[int]]:
    crossover_index  = np.random.randint(1, len(solution_1))
    child_1: list[int] = copy.deepcopy(solution_1)
    child_2: list[int] = copy.deepcopy(solution_2)
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
    mutation: Callable[[list[int]], list[int]] = random.choice([swap_mutation, inversion_mutation, scramble_mutation])
    return mutation(solution)

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

def test_first_better_neighbor():
    print("Hill Climbing: (First better neighbor):")
    solution, cost = hill_climbing(test_packages, get_first_better_neighbor, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 1000")
    solution, cost = hill_climbing(test_packages, get_first_better_neighbor, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 10000")
    solution, cost = hill_climbing(test_packages, get_first_better_neighbor, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 100000")
    print()

def test_best_neighbor():
    print("Hill Climbing (Best neighbor):")
    solution, cost = hill_climbing(test_packages, get_best_neighbor, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 1000")
    solution, cost = hill_climbing(test_packages, get_best_neighbor, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 10000")
    solution, cost = hill_climbing(test_packages, get_best_neighbor, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 100000")
    print()

def test_random_neighbor():
    print("Hill Climbing (Random neighbor):")
    solution, cost = hill_climbing(test_packages, get_random_neighbor, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 1000")
    solution, cost = hill_climbing(test_packages, get_random_neighbor, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 10000")
    solution, cost = hill_climbing(test_packages, get_random_neighbor, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 100000")
    print()

def test_simulated_annealing():
    print("Simulated Annealing:")
    solution, cost = simulated_annealing(test_packages, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 1000")
    solution, cost = simulated_annealing(test_packages, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 10000")
    solution, cost = simulated_annealing(test_packages, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 100000")
    solution, cost = simulated_annealing(test_packages, iterations=1000000)
    print(f"{solution} -> {round(cost, 2)} -> iterations = 1000000")
    print()

def test_tabu_search():
    print("Tabu Search:")
    solution, cost = tabu_search(test_packages, tabu_list_size=100, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 100 iterations = 1000")
    solution, cost = tabu_search(test_packages, tabu_list_size=1000, iterations=1000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 1000 iterations = 1000")
    solution, cost = tabu_search(test_packages, tabu_list_size=100, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 100 iterations = 10000")
    solution, cost = tabu_search(test_packages, tabu_list_size=1000, iterations=10000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 1000 iterations = 10000")
    solution, cost = tabu_search(test_packages, tabu_list_size=100, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 100 iterations = 100000")
    solution, cost = tabu_search(test_packages, tabu_list_size=1000, iterations=100000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 1000 iterations = 100000")
    solution, cost = tabu_search(test_packages, tabu_list_size=100, iterations=1000000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 100 iterations = 1000000")
    solution, cost = tabu_search(test_packages, tabu_list_size=1000, iterations=1000000)
    print(f"{solution} -> {round(cost, 2)} -> tabu_list_size = 1000 iterations = 1000000")
    print()

if __name__ == "__main__":
    
    test_first_better_neighbor()
    test_best_neighbor()
    test_random_neighbor()
    test_simulated_annealing()
    test_tabu_search()

