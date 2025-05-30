# Solves the deterministic SSCFLP with fixed demands and penalty.
# Evaluates its performance on stochastic model with given number of scenarios.
# python SSCFLP.py data_name nr_scenarios

import gurobipy as gp
from gurobipy import GRB
import sys
import random
import numpy as np
import time

random.seed(42)
PEN=1251768

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        num_locations, num_customers = map(int, file.readline().split())

        locations = []
        for _ in range(num_locations):
            capacity, fixed_costs = map(float, file.readline().split())
            locations.append({'capacity': capacity, 'fixed_costs': fixed_costs})

        customer_demands = []
        while len(customer_demands) < num_customers:
            customer_demands.extend(map(float, file.readline().split()))

        distance_matrix = []
        for line in file:
            if not line.strip() or not all(is_numeric(value) for value in line.split()):
                break
            distance_matrix.append(list(map(float, line.split())))

        # Check if distance matrix needs transposation
        if (len(distance_matrix) == num_customers) and (len(distance_matrix[0]) == num_locations):
            distance_matrix = list(map(list, zip(*distance_matrix)))

    return num_locations, num_customers, locations, customer_demands, distance_matrix

def solve_SSCFLP(num_locations, num_customers, locations, customer_demands, distance_matrix):
    model = gp.Model("SSCFLP")

    x = model.addVars(num_locations, vtype=GRB.BINARY, name="x") 
    y = model.addVars(num_locations, num_customers, vtype=GRB.BINARY, name="y")  
    u = model.addVars(num_customers, vtype=GRB.BINARY, name="u") 

    model.setObjective(
        gp.quicksum(locations[i]['fixed_costs'] * x[i] for i in range(num_locations)) + 
        gp.quicksum(distance_matrix[i][j] * y[i, j] for i in range(num_locations) for j in range(num_customers)) +
        gp.quicksum(PEN * u[j] for j in range(num_customers)),
        sense=GRB.MINIMIZE
    )

    model.addConstrs((gp.quicksum(y[i, j] for i in range(num_locations)) == 1 - u[j] for j in range(num_customers)), name="assign_customers")
    model.addConstrs((gp.quicksum(y[i, j] * customer_demands[j] for j in range(num_customers)) <= locations[i]['capacity'] * x[i] for i in range(num_locations)), name="facility_capacity")

    model.setParam('OutputFlag', 0) 
    model.optimize()

    open_facilities=[]

    if model.status == GRB.OPTIMAL:        
        res = model.objVal
#        cost = 0
        for i in range(num_locations):
            if x[i].x > 0.5:
                open_facilities.append(i+1) 
#            cost = cost+locations[i]['fixed_costs']*x[i].x
    return(open_facilities,res)  

def add_stochasticity(value):
    percentage_change = random.uniform(-0.30, 0.30)
    return round(value * (1 + percentage_change), 2)

def solve_recourse(first_stage_solution, distance_matrix, s_demands):
    num_locations = len(first_stage_solution)
    num_customers = len(s_demands)

    model = gp.Model("Second_Stage_Problem")
    y = model.addVars(num_locations, num_customers, vtype=GRB.BINARY, name="y")
    u = model.addVars(num_customers, vtype=GRB.BINARY, name="u")  

    model.setObjective(
        gp.quicksum(distance_matrix[first_stage_solution[i]-1][j] * y[i, j] for i in range(num_locations) for j in range(num_customers)) +
        gp.quicksum(PEN* u[j] for j in range(num_customers)),
        sense=GRB.MINIMIZE
    )

    model.addConstrs((gp.quicksum(y[i, j] for i in range(num_locations)) == 1 - u[j] for j in range(num_customers)), name="assign_customers")
    model.addConstrs((gp.quicksum(y[i, j] * s_demands[j] for j in range(num_customers)) <= locations[first_stage_solution[i]-1]['capacity'] for i in range(num_locations)), name="facility_capacity")

    model.setParam('OutputFlag', 0)
    model.optimize()
    if model.status == GRB.OPTIMAL:  
        res = model.objVal
    return res 


def MC_evaluator(first_stage_solution):
    second = []
    first = sum(locations[i - 1]['fixed_costs'] for i in first_stage_solution)
    for i in range(nr_scenarios):
        stochastic_demands = [add_stochasticity(value) for value in customer_demands]
        second.append(solve_recourse(first_stage_solution,distance_matrix,stochastic_demands))
    return first + sum(second)/nr_scenarios

start_time = time.time()
file_path = sys.argv[1]
nr_scenarios = int(sys.argv[2])
num_locations, num_customers, locations, customer_demands, distance_matrix = read_dataset(file_path)
first_stage_sol,det_obj= solve_SSCFLP(num_locations, num_customers, locations, customer_demands, distance_matrix)
res = MC_evaluator(first_stage_sol)
end_time = time.time()
print(file_path,round(end_time - start_time,2),num_locations,num_customers,first_stage_sol, det_obj,res)
