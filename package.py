import random
import pandas as pd
import numpy as np
import copy
from typing import Callable

# Constraints
cost_per_km = penalty_cost_per_minute = 0.3
speed = 60

reputation_weight = 0.5

num_packages = 10
map_size = 60


class Package:
    def __init__(self, package_type, coordinates):
        self.package_type: str = package_type
        self.coordinates_x: int = coordinates[0]
        self.coordinates_y: int = coordinates[1]
        if package_type == 'fragile':
            self.breaking_chance = random.uniform(0.0001, 0.01) # 0.01-1%chance of breaking per km
            self.breaking_cost = random.uniform(3, 10) # Extra cost in case of breaking
        elif package_type == 'urgent':
            self.delivery_time = random.uniform(100, 240) # Delivery time in minutes (100 minutes to 4 hours)
            
    def generate_package_stream(num_packages, map_size):
        package_types = ['fragile', 'normal', 'urgent']
        package_stream = [Package(random.choice(package_types),(random.uniform(0, map_size), random.uniform(0, map_size))) for _ in range(num_packages)]
        return package_stream
    
    def __str__(self):
        return f"Package type: {self.package_type}, coordinates: ({self.coordinates_x}, {self.coordinates_y})"
    
    def distance(self, coordinates):
        return ((self.coordinates_x - coordinates[0])**2 + (self.coordinates_y - coordinates[1])**2)**0.5

def is_damaged(package: Package, distance_covered: float) -> bool:
    p_damage = 1 - (1 - package.breaking_chance)**distance_covered
    return random.uniform(0, 1) < p_damage



def generate_random_solution():
    return np.random.choice(range(num_packages), num_packages, replace=False)

def calculate_unexpected_cost(package: Package, distance_covered: float, penalty_cost_per_minute: float) -> float:
    cost: float = 0

    if package.package_type == 'urgent' and package.delivery_time < distance_covered / speed * 60:
        penalty = (distance_covered / speed * 60 - package.delivery_time) * penalty_cost_per_minute
        cost -= penalty

    if package.package_type == 'fragile' and is_damaged(package, distance_covered):
        cost += package.breaking_cost

    return cost


def evaluate_solution(solution: list[int], packages: list[Package], cost_per_km: float, penalty_cost_per_minute: float) -> float:
    total_cost: float = 0
    distance_covered: float = 0
    reputation_cost: float = 0

    distance_covered += packages[solution[0]].distance((0, 0))
    total_cost += distance_covered * cost_per_km + calculate_unexpected_cost(packages[solution[0]], distance_covered, penalty_cost_per_minute)

    for i in range(1, len(solution)):
        distance_covered += packages[solution[i]].distance((packages[solution[i - 1]].coordinates_x, packages[solution[i - 1]].coordinates_y))
        distance_cost = distance_covered * cost_per_km
        reputation_cost += calculate_unexpected_cost(packages[solution[i]], distance_covered, penalty_cost_per_minute)
        
        total_cost += (1 - reputation_weight) * distance_cost + reputation_weight * reputation_cost
    return total_cost

# hill climbing

# neighbour functions 

# switches 2 random packages in the final solution
def get_neighbor_solution(solution: list[int]) -> list[int]:
    neighbor_solution = copy.deepcopy(solution)
    i, j = random.sample(range(len(solution)), 2)
    neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
    return neighbor_solution


def hill_climbing(packages: list[Package], cost_per_km: float, penalty_cost_per_minute: float, neighbor_function: Callable [[list[int]], list[int]], iterations: int = 1000) -> list[int]:
    num_packages = len(packages)
    best_solution = random.sample(range(num_packages), num_packages)
    best_cost = evaluate_solution(best_solution, packages, cost_per_km, penalty_cost_per_minute)

    for _ in range(iterations):
        solution = neighbor_function(best_solution)
        cost = evaluate_solution(solution, packages, cost_per_km, penalty_cost_per_minute)

        if cost < best_cost:
            best_cost = cost
            best_solution = solution

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
    print("Crossover Point: ", crossover_index)
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


def inversion_mutation(solution):
    index_1 = np.random.randint(0, len(solution))
    index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    solution[index_1:index_2] = solution[index_1:index_2][::-1]
    return solution

def scramble_mutation(solution):
    index_1 = np.random.randint(0, len(solution))
    index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1)
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    np.random.shuffle(solution[index_1:index_2])
    return solution

def random_mutation(solution):
    mutation = np.random.choice([swap_mutation, inversion_mutation, scramble_mutation])
    return mutation(solution)

# selection functions

#tournament solution
def greatest_fit(population: list[list[int]], packages: list[Package], cost_per_km: float, penalty_cost_per_minute: float):  # podemos ter de retornar a score para além da solução
    best_solution = population[0]
    best_score = evaluate_solution(best_solution, packages, cost_per_km, penalty_cost_per_minute)
    for sol in population:
        score = evaluate_solution(sol, packages, cost_per_km, penalty_cost_per_minute)
        if score > best_score:
            best_solution = sol
            best_score = score
    return best_solution
    

def generate_population(population_size):
    solutions = []
    for _ in range(population_size):
        solutions.append(generate_random_solution())
    return solutions

def tournament_selection(population, packages, cost_per_km, penalty_cost_per_minute, tournament_size=20):
    if len(population) < tournament_size:
        return greatest_fit(population, packages, cost_per_km, penalty_cost_per_minute)
    participants = random.sample(population, tournament_size)
    return greatest_fit(participants, packages, cost_per_km, penalty_cost_per_minute)
    
# roulette wheel selection
def roulette_wheel_selection(population: list[list[int]], packages: list[Package], cost_per_km: float, penalty_cost_per_minute: float):
    scores = [evaluate_solution(sol, packages, cost_per_km, penalty_cost_per_minute) for sol in population]
    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]
    return population[np.random.choice(len(population), p=probabilities)]

def random_selection(population, packages, cost_per_km, penalty_cost_per_minute):
    selection_function = np.random.choice([tournament_selection, roulette_wheel_selection])
    if selection_function == tournament_selection:
        return selection_function(population, packages, cost_per_km, penalty_cost_per_minute, tournament_size=20)
    else:
        return selection_function(population, packages, cost_per_km, penalty_cost_per_minute)


def genetic_algorithm(packages : list[Package], cost_per_km: float, penalty_cost_per_minute: float,crossover_function, population_size=100, generations=10, mutation_rate=0.01):
    population = generate_population(population_size)
    for _ in range(generations):
        new_population = []
        for i in range(1, population_size):
            parent1 = random_selection(population, packages, cost_per_km, penalty_cost_per_minute)
            parent2 = random_selection(population, packages, cost_per_km, penalty_cost_per_minute)
            child1, child2 = crossover_function(parent1, parent2)
            if np.random.rand() < mutation_rate:
                child1 = random_mutation(child1)
            # if np.random.rand() < mutation_rate:
            #     child2 = random_mutation(child2)
            new_population.append(child1)
            # new_population.append(child2)

        population = new_population

    solution = greatest_fit(population, packages, cost_per_km, penalty_cost_per_minute)
    cost = evaluate_solution(solution, packages, cost_per_km, penalty_cost_per_minute)
    return solution, cost


# Example: Generate a stream of 15 packages in a map of size 60x60
package_stream = Package.generate_package_stream(num_packages, map_size)


df = pd.DataFrame([(i, package.package_type, package.coordinates_x,
package.coordinates_y, package.breaking_chance if package.package_type ==
'fragile' else None, package.breaking_cost if package.package_type ==
'fragile' else None, package.delivery_time if package.package_type ==
'urgent' else None) for i, package in enumerate(package_stream, start=1)],
columns=["Package", "Type", "CoordinatesX", "CoordinatesY", "Breaking Chance", "Breaking Cost", "Delivery Time"])


#print(df.to_string(index=False))
print()

solution, cost = genetic_algorithm(package_stream, cost_per_km, penalty_cost_per_minute, pmx_crossover)
print(f"Best solution: {solution}")
print(f"Best cost: {cost}")
# solution1 = generate_random_solution()
# solution2 = generate_random_solution()
# print(solution1)
# print(solution2)

# child_1, child_2 = pmx_crossover(solution1, solution2)
# print (child_1)
# print (child_2)