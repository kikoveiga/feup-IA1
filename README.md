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

* **Maintain a high reputation** by keeping packages from breaking and delivering urgent packages on time.
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

###  Solution Representation

The solution can be represented as a list of integers representing the indexes of each package on the original list of deliveries.
