# IA1

## Problem Definition
specification of the work to be performed (definition of the game or optimization problem to be solved)

This project has as its objective to optimize the delivery of packages from a starting point (0,0) to various destinations, optimizing the travel costs while adhering to each package type's special needs.
It can be succintly described as follow.

### Scenario

There are 3 different types of packages: fragile, normal and urgent.

* Fragile packages have a change of getting damaged by kilometer traveled.
* Urgent packages have a penalty associated with delays.
* Each kilometer traveled incurs a fixed cost.

### Objective

For an algorithm to be considered successful it needs to:

* **Maintain a high repution** by keeping packages from breaking and delivering urgent packages on time.
* **Reduce the total cost** which includes penalties from late deliveries, packages broken and distance travalled.


### Constraints

For this project the world will always follow some rules:

* Only one vehicle is available.
* Delivery locations are specified by a pair of coordinates (x,y)
* There are always direct routes between all delivery places.
* The vehicle travels at a constant speed of 60km/hour
* The cost per kilometer is fixed
* The delivery staff takes 0 seconds to deliver each package
  

## References / Src code
related work with references to works found in a bibliographic search (articles, web pages and/or source code)

## Search Problem
formulation of the problem as a search problem (state representation, initial state, objective test, operators (names, preconditions, effects and costs), heuristics/evaluation function) or optimization problem (solution representation, neighborhood/mutation and crossover functions, hard constraints, evaluation functions)

###  State Representation

**State Representation**(T,P,D,C,U)
* T = Truck coordinates
* P = List of undelivered packages
* D = Distance traveled
* C = Cost so far
* U = Penalties

**Initial State** :  ((0,0), P, 0, 0, 0)
In the beggining the truck is in the position (0,0), all packages are still to be delivered , the distance traveled is 0, there are still no late urgent deliveries, and therefore no cost.

Since there is no advantage in moving the truck without delivering any package, the only operator needed is the operator move, which can be defined as following.

**Name:** Move
**Preconditions:** 
* There are undelivered packages remaining

**Effects**
* Updates the Truck coordinates in the state
* Removes the delivered package from the undelivered packages list
* Increases the total distance travalled and associated cost
* Increases the penalties cost if the delivered package is fragile and is broken.
* Increases the penalties cost if any urgent packages is still to be delivered after schedule.

**Cost** 


