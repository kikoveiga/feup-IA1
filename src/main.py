from menu import display_menu, hill_climbing_menu, simulated_annealing_menu, tabu_search_menu, genetic_algorithm_menu, settings_menu

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            hill_climbing_menu()
        elif choice == "2":
            simulated_annealing_menu()
        elif choice == "3":
            tabu_search_menu()
        elif choice == "4":
            genetic_algorithm_menu()
        elif choice == "5":
            settings_menu()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()