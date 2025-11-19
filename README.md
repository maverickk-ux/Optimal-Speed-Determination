#Optimal Speed and Fuel Cost Minimization Using Optimization Algorithms

This project implements and compares classical optimization methods to determine the optimal driving speed that minimizes total travel cost, which includes fuel cost and time cost.
The model incorporates real-world parameters such as traffic conditions, number of passengers, road gradient, engine efficiency, and fuel price.

This project was developed as part of an Optimization Course Project.

#Project Overview

Commuters face a trade-off between fuel efficiency and travel time.
Driving too slowly increases travel time, while driving too fast increases fuel consumption due to aerodynamic drag.

This project models that trade-off and uses optimization techniques to find the speed 
𝑣
v (in km/h) that results in the minimum total trip cost.

⚙️ Optimization Algorithms Used

The following algorithms are implemented and compared:

Steepest Descent (Gradient Descent)

Gradient Descent with Line Search

Newton’s Method

Quasi-Newton Method (BFGS)

Trust-Region Method

Each method is evaluated based on:

Convergence speed

Number of iterations

Stability

Effectiveness across different traffic and passenger conditions
