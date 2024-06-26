import random, pandas as pd
from constraints import map_size, num_packages, speed, cost_per_km, penalty_cost_per_minute, reputation_weight

class Package:
    def __init__(self: 'Package', package_type: str, coordinates: tuple[float, float]):
        self.package_type: str = package_type
        self.coordinates_x: float = coordinates[0]
        self.coordinates_y: float = coordinates[1]
        if package_type == 'fragile':
            self.breaking_chance = random.uniform(0.0001, 0.01) # 0.01-1%chance of breaking per km
            self.breaking_cost = random.uniform(3, 10) # Extra cost in case of breaking
        elif package_type == 'urgent':
            self.delivery_time = random.uniform(100, 240) # Delivery time in minutes (100 minutes to 4 hours)
    
    def is_damaged(self: 'Package', distance_covered: float) -> bool:
        p_damage = 1 - (1 - self.breaking_chance)**distance_covered
        return random.uniform(0, 1) < p_damage
    
    def calculate_unexpected_cost(self: 'Package', distance_covered: float, penalty_cost_per_minute: float) -> float:
        cost: float = 0

        if self.package_type == 'urgent' and self.delivery_time < distance_covered / speed * 60:
            penalty = (distance_covered / speed * 60 - self.delivery_time) * penalty_cost_per_minute
            cost += penalty

        if self.package_type == 'fragile' and self.is_damaged(distance_covered):
            cost += self.breaking_cost

        return cost
    
    def __str__(self):
        return f"Package type: {self.package_type}, coordinates: ({self.coordinates_x}, {self.coordinates_y})"
    
    def distance(self, coordinates):
        return ((self.coordinates_x - coordinates[0])**2 + (self.coordinates_y - coordinates[1])**2)**0.5

    def __repr__(self):
        return f"Package('{self.package_type}', ({self.coordinates_x}, {self.coordinates_y}))"
    
    
def generate_package_stream(num_packages: int = num_packages, map_size: int = map_size) -> list[Package]:
    package_types = ['fragile', 'normal', 'urgent']
    package_stream = [Package(random.choice(package_types),(random.uniform(0, map_size), random.uniform(0, map_size))) for _ in range(num_packages)]
    return package_stream

def generate_random_solution() -> list[int]:
    return random.sample(range(num_packages), num_packages)

def evaluate_solution(package_stream: list[Package], solution: list[int]) -> float:
    total_cost: float = 0
    distance_covered: float = 0
    reputation_cost: float = 0

    distance_covered += package_stream[solution[0]].distance((0, 0))
    total_cost += (1 - reputation_weight) * distance_covered * cost_per_km + reputation_weight * package_stream[solution[0]].calculate_unexpected_cost(distance_covered, penalty_cost_per_minute)

    for i in range(1, len(solution)):
        distance_covered += package_stream[solution[i]].distance((package_stream[solution[i - 1]].coordinates_x, package_stream[solution[i - 1]].coordinates_y))
        distance_cost = distance_covered * cost_per_km
        reputation_cost += package_stream[solution[i]].calculate_unexpected_cost(distance_covered, penalty_cost_per_minute)
        
        total_cost += (1 - reputation_weight) * distance_cost + reputation_weight * reputation_cost
    return total_cost

if __name__ == "__main__":

    print(generate_random_solution())

    package_stream = generate_package_stream(num_packages, map_size)
    solution = generate_random_solution()
    for _ in range(10):
        print(round(evaluate_solution(package_stream, solution), 2))


    df = pd.DataFrame([(i, package.package_type, package.coordinates_x, package.coordinates_y,
                        package.breaking_chance if package.package_type == 'fragile' else None,
                        package.breaking_cost if package.package_type == 'fragile' else None,
                        package.delivery_time if package.package_type == 'urgent' else None)
                        for i, package in enumerate(package_stream, start=1)],
        columns=["Package", "Type", "CoordinatesX", "CoordinatesY", "Breaking Chance", "Breaking Cost", "Delivery Time"])

    print()
    print(df.to_string(index=False))
