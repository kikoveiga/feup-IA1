import random
import pandas as pd
import numpy as np

# Constraints
cost_per_km = penalty_cost_per_minute = 0.3
speed = 60


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

    distance_covered += packages[solution[0]].distance((0, 0))
    total_cost += distance_covered * cost_per_km + calculate_unexpected_cost(packages[solution[0]], distance_covered, penalty_cost_per_minute)

    for i in range(1, len(solution)):
        distance_covered += packages[solution[i]].distance((packages[solution[i - 1]].coordinates_x, packages[solution[i - 1]].coordinates_y))
        total_cost += distance_covered * cost_per_km + calculate_unexpected_cost(packages[solution[i]], distance_covered, penalty_cost_per_minute)
    
    return total_cost

def genetic_algorithm(packages, cost_per_km: float, penalty_cost_per_minute: float, population_size=100, generations=1000, mutation_rate=0.01):
    num_packages = len(packages)
    best_solution = None
    best_cost = float('inf')

    for _ in range(generations):
        population = [random.sample(range(num_packages), num_packages) for _ in range(population_size)]

        for i in range(population_size):
            solution = population[i]
            cost = evaluate_solution(solution, packages, cost_per_km, penalty_cost_per_minute)

            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        population.sort(key=lambda x: evaluate_solution(x, packages, cost_per_km, penalty_cost_per_minute))
        elite_size = int(0.2 * population_size)
        next_generation = population[:elite_size]

        for _ in range(population_size - elite_size):
            parent1 = random.choice(population[:elite_size])
            parent2 = random.choice(population[:elite_size])
            crossover_point = random.randint(0, num_packages - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            next_generation.append(child)

        for i in range(elite_size, population_size):
            if random.uniform(0, 1) < mutation_rate:
                mutation_point1 = random.randint(0, num_packages - 1)
                mutation_point2 = random.randint(0, num_packages - 1)
                next_generation[i][mutation_point1], next_generation[i][mutation_point2] = \
                    next_generation[i][mutation_point2], next_generation[i][mutation_point1]

        population = next_generation

    return best_solution, best_cost

# Example: Generate a stream of 15 packages in a map of size 60x60
num_packages = 15
map_size = 60
package_stream = Package.generate_package_stream(num_packages, map_size)

df = pd.DataFrame([(i, package.package_type, package.coordinates_x,
package.coordinates_y, package.breaking_chance if package.package_type ==
'fragile' else None, package.breaking_cost if package.package_type ==
'fragile' else None, package.delivery_time if package.package_type ==
'urgent' else None) for i, package in enumerate(package_stream, start=1)],
columns=["Package", "Type", "CoordinatesX", "CoordinatesY", "Breaking Chance", "Breaking Cost", "Delivery Time"])


print(df.to_string(index=False))
print()

# Example: Run the genetic algorithm
solution, cost = genetic_algorithm(package_stream, cost_per_km, penalty_cost_per_minute)
print(f"Best solution: {solution}")
print(f"Best cost: {cost}")