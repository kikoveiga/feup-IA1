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

class icons:
    TOP_LEFT = '\u250f'
    TOP_RIGHT = '\u2513'
    BOTTOM_LEFT = '\u2517'
    BOTTOM_RIGHT = '\u251b'
    VERTICAL = '\u2503'
    HORIZONTAL = '\u2501'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

    def __repr__(self):
        return f"Package('{self.package_type}', ({self.coordinates_x}, {self.coordinates_y}))"
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


def evaluate_solution(solution: list[int]) -> float:
    total_cost: float = 0
    distance_covered: float = 0
    reputation_cost: float = 0

    distance_covered += package_stream[solution[0]].distance((0, 0))
    total_cost += distance_covered * cost_per_km + calculate_unexpected_cost(package_stream[solution[0]], distance_covered, penalty_cost_per_minute)

    for i in range(1, len(solution)):
        distance_covered += package_stream[solution[i]].distance((package_stream[solution[i - 1]].coordinates_x, package_stream[solution[i - 1]].coordinates_y))
        distance_cost = distance_covered * cost_per_km
        reputation_cost += calculate_unexpected_cost(package_stream[solution[i]], distance_covered, penalty_cost_per_minute)
        
        total_cost += (1 - reputation_weight) * distance_cost + reputation_weight * reputation_cost
    return total_cost

# function to evaluate the cost of a neighbour solution more efficiently than evaluating the whole solution again
#def evaluate_neighbour_solution(solution: list[int], cost: int, index_1: int, index_2: int) -> float:

# hill climbing

# neighbour functions 

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




def print_packages_on_map(packages):
    print (f"Number of packages: {num_packages}")
    print (f"Size: {map_size}x{map_size}")
    converted_map_size = 30
    char_width = 2  # Adjust for the character width being half the character height

    map_grid = [[' ' for _ in range(converted_map_size * char_width)] for _ in range(converted_map_size)]
    map_grid[converted_map_size // 2][converted_map_size // 2 * char_width] = 'X'  # Starting point
    for i, package in enumerate(packages):
        x, y = package.coordinates_x, package.coordinates_y
        x = int(x * (converted_map_size / map_size))  # Adjust x-coordinate to fit 30x30 map
        y = int(y * (converted_map_size / map_size))  # Adjust y-coordinate to fit 30x30 map

        if package.package_type == 'urgent':
            symbol = bcolors.FAIL + str(i) + bcolors.ENDC
        elif package.package_type    == 'fragile':
            symbol = bcolors.OKBLUE + str(i) + bcolors.ENDC
        else:
            symbol = str(i)

        if 0 <= x < converted_map_size and 0 <= y < converted_map_size:
            map_grid[y][x * char_width] = symbol  # Adjusting for character width

    # Top border
    print(icons.TOP_LEFT + icons.HORIZONTAL * (converted_map_size * char_width) + icons.TOP_RIGHT)

    for row in map_grid:
        print(icons.VERTICAL, end='')
        for cell in row:
            print(cell, end='')
        print(icons.VERTICAL)

    # Bottom border
    print(icons.BOTTOM_LEFT + icons.HORIZONTAL * (converted_map_size * char_width) + icons.BOTTOM_RIGHT)

    # Legend
    print(f"{bcolors.FAIL}Urgent{bcolors.ENDC} | {bcolors.OKBLUE}Fragile{bcolors.ENDC} | Normal")
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
test_packages = [Package('urgent', (56.97556558957435, 27.90045109824626)),
                Package('fragile', (44.8203921010497, 45.239271577485084)),
                Package('normal', (3.23756109561091, 21.1712638842833)),
                Package('normal', (31.342549842605905, 43.79155490345392)),
                Package('normal', (54.51149696067019, 55.917516798837994)),
                Package('normal', (0.13982076073304261, 14.455164572152892)),
                Package('fragile', (59.311394005015615, 52.17887506703184)),
                Package('fragile', (43.07637979465492, 7.6826233225053775)),
                Package('fragile', (43.662203998731016, 33.37818643749602)),
                Package('urgent', (23.530958559725182, 3.3941997983349426))]

package_stream = test_packages

show_map = True

def print_menu_title_box(title):
    print("\n\n")
    print(icons.TOP_LEFT + icons.HORIZONTAL * (len(title) + 2) + icons.TOP_RIGHT)
    print(icons.VERTICAL + " " + title + " " + icons.VERTICAL)
    print(icons.BOTTOM_LEFT + icons.HORIZONTAL * (len(title) + 2) + icons.BOTTOM_RIGHT)
    print()

def hill_climbing_menu():
    print_menu_title_box("Hill Climbing Menu")
    iterations = int(input("Desired Number of iterations: "))

    if show_map:
        print_packages_on_map(package_stream)
    solution, cost = hill_climbing(get_neighbor_solution, iterations)
    print(f"Best Solution: {solution}")
    print(f"Cost: {cost}")

def simulated_annealing_menu():
    print_menu_title_box("Simulated Annealing Menu")
    iterations = int(input("Number of Iterations: "))
    temperature = float(input("Initial temperature: "))
    cooling = float(input("Cooling Rate: "))
    if show_map:
        print_packages_on_map(package_stream)
    # Call simulated annealing algorithm function with specified parameters
    solution = simulated_annealing(get_neighbor_solution, iterations, temperature, cooling)
    print(f"Best Solution: {solution}")
    print(f"Cost: {evaluate_solution(solution)}")

def genetic_algorithm_menu():
    print_menu_title_box("Genetic Algorithm Menu")
    generations = int(input("Number of generations: "))
    population_size = int(input("Population Size: "))
    mutation_rate = float(input("Mutation Rate: "))

    if show_map:
        print_packages_on_map(package_stream)
    # Call genetic algorithm function with specified parameters
    solution, cost = genetic_algorithm(pmx_crossover, population_size, generations, mutation_rate)
    print(f"Best Solution: {solution}")
    print(f"Cost: {cost}")


def settings_menu():
    while True:
        print_menu_title_box("Settings Menu")
        print("1. Change Map Size")
        print("2. Change Number of Packages")
        print("3. Toggle Show Map")
        print("4. Back to Main Menu")
        choice = input("Enter your choice: ")

        if choice == "1":
            change_map_size()
        elif choice == "2":
            change_num_packages()
        elif choice == "3":
            toggle_show_map()
        elif choice == "4":
            print("Returning to Main Menu...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

def toggle_show_map():
    global show_map
    show_map = not show_map
    print(f"Show map updated to {show_map}.")

def change_map_size():
    global map_size
    new_size = int(input("Enter the new map size: "))
    if new_size > 0:
        map_size = new_size
        print("Map size updated successfully.")
    else:
        print("Invalid map size. Please enter a positive integer.")

def change_num_packages():
    global num_packages
    new_packages = int(input("Enter the new number of packages: "))
    if new_packages > 0:
        num_packages = new_packages
        print("Number of packages updated successfully.")
    else:
        print("Invalid number of packages. Please enter a positive integer.")

def display_menu():
    print_menu_title_box("Main Menu")
    print("Welcome to the Package Delivery System:")
    print("1. Run Hill Climbing Algorithm")
    print("2. Run Simulated Annealing Algorithm")
    print("3. Run Genetic Algorithm")
    print("4. Settings")
    print("5. Exit")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            hill_climbing_menu()
        elif choice == "2":
            simulated_annealing_menu()
        elif choice == "3":
            genetic_algorithm_menu()
        elif choice == "4":
            settings_menu()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()


# print_packages_on_map(package_stream)

# Test Hill Climbing
# print("Hill Climbing:")
# solution, cost = hill_climbing(get_neighbor_solution, iterations=1000000)

# Test SA
#solution, cost = simulated_annealing(get_neighbor_solution)

# Test Genetic
#for _ in range(5):
#    solution, cost = genetic_algorithm(pmx_crossover, generations=1000, population_size=50)
#    print(f"Best solution: {solution}")
#    print(f"Best cost: {cost}")


# print(f"Best solution: {solution}")
# print(f"Best cost: {cost}")