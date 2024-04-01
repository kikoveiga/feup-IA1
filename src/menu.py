from package import Package, generate_package_stream
from algorithms import hill_climbing, simulated_annealing, tabu_search, genetic_algorithm, pmx_crossover, get_random_neighbor, evaluate_solution, get_best_neighbor
from constraints import num_packages, map_size

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

package_stream: list[Package] = test_packages
show_map: bool = True

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

def print_packages_on_map(packages):
    print (f"Number of packages: {num_packages}")
    print (f"Size: {map_size}x{map_size}")
    converted_map_size = 30
    char_width = 2  # Adjust for the character width being half the character height

    map_grid = [[' ' for _ in range(converted_map_size * char_width)] for _ in range(converted_map_size)]
    map_grid[converted_map_size // 2][converted_map_size // 2 * char_width] = f"{bcolors.OKGREEN}X{bcolors.ENDC}"  # Starting point
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
    solution, cost = hill_climbing(package_stream, get_best_neighbor, iterations)
    print(f"Best Solution: {solution}")
    print(f"Cost: {round(cost, 2)}")

def simulated_annealing_menu():
    print_menu_title_box("Simulated Annealing Menu")
    iterations = int(input("Number of Iterations: "))
    temperature = float(input("Initial temperature: "))
    cooling = float(input("Cooling Rate: "))
    if show_map:
        print_packages_on_map(package_stream)
    # Call simulated annealing function with specified parameters
    solution, cost = simulated_annealing(package_stream, iterations, temperature, cooling)
    print(f"Best Solution: {solution}")
    print(f"Cost: {round(cost, 2)}")

def tabu_search_menu():
    print_menu_title_box("Tabu Search Menu")
    tabu_list_size = int(input("Tabu List Size: "))
    iterations = int(input("Number of Iterations: "))
    if show_map:
        print_packages_on_map(package_stream)
    # Call tabu search function with specified parameters
    solution, cost = tabu_search(package_stream, tabu_list_size, iterations)
    print(f"Best Solution: {solution}")
    print(f"Cost: {round(cost, 2)}")

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
    print(f"Cost: {round(cost, 2)}")


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
    print("1. Run Hill Climbing (with Best Neighbor)")
    print("2. Run Simulated Annealing")
    print("3. Run Tabu Search")
    print("4. Run Genetic Algorithm")
    print("5. Settings")
    print("6. Exit")