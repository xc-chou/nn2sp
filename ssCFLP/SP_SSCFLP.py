# Solves Deterministic Equivalent of SP with pyomo + mpisppy
# python sp.py data_name nr_scenarios

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
import sys
import numpy as np
import random
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

        # Read distance matrix
        distance_matrix = []

        for line in file:
            if not line.strip() or not all(is_numeric(value) for value in line.split()):
                break
            distance_matrix.append(list(map(float, line.split())))

        # Check if distance matrix needs transposation
        if (len(distance_matrix) == num_customers) and (len(distance_matrix[0]) == num_locations):
            distance_matrix = list(map(list, zip(*distance_matrix)))

    return num_locations, num_customers, locations, customer_demands, distance_matrix

file_path = sys.argv[1]
num_locations, num_customers, locations, customer_demands, distance_matrix = read_dataset(file_path)
max_capacity = max(location['capacity'] for location in locations)
nr_scenarios = int(sys.argv[2])

def build_model(demands):
    model = pyo.ConcreteModel()

    model.locations = set(range(num_locations))
    model.customers = set(range(num_customers))
    model.x = pyo.Var(model.locations, domain=pyo.Binary)
    model.y = pyo.Var(model.locations, model.customers, domain=pyo.Binary)
    model.p = pyo.Var(model.customers,domain=pyo.Binary)

    model.firstStageCost = sum(model.x[i] * locations[i]['fixed_costs'] for i in model.locations)
    model.secondStageCost = sum(distance_matrix[i][j] * model.y[i, j] for i in model.locations for j in model.customers) + sum(PEN * model.p[j] for j in model.customers)

    model.OBJ = pyo.Objective(
        expr=model.firstStageCost + model.secondStageCost,
        sense=pyo.minimize
    )

    model.CONSTR = pyo.ConstraintList()
    for j in range(num_customers):
        model.CONSTR.add(sum(model.y[i, j] for i in model.locations) == 1 - model.p[j])
    for i in range(num_locations):
        model.CONSTR.add(sum(model.y[i, j] * demands[j] for j in model.customers) <= locations[i]['capacity'] * model.x[i])

    return model

#demands = customer_demands
#model = build_model(demands)

#solver = pyo.SolverFactory('gurobi')
#results = solver.solve(model)

#obj_value = pyo.value(model.OBJ)
#print(f"Objective Value: {obj_value}")
#for i in model.locations:
#    if pyo.value(model.x[i]) > 0.5:
#        served_customers = [j+1 for j in range(num_customers) if pyo.value(model.y[i, j]) > 0]
#        print(i+1, served_customers)
#det_fss = []
#for i in model.locations:
#    if pyo.value(model.x[i]) > 0.5:
#        det_fss.append(i+1)
#print(det_fss, obj_value,'Deterministic problem')

# Stochastic version
max_capacity = max(location['capacity'] for location in locations)
def add_stochasticity(value):
    percentage_change = random.uniform(-0.30, 0.30)
    return round(value * (1 + percentage_change), 2)

def scenario_creator(scenario_name):
    for i in range(nr_scenarios):
        if scenario_name == "s"+str(i):
            sdemands = [add_stochasticity(value) for value in customer_demands]
    model = build_model(sdemands)

    sputils.attach_root_node(model, model.firstStageCost, [model.x])
    model._mpisppy_probability = 1.0 / nr_scenarios
    return model

# Extensive Form
start_time = time.time()
options = {"solver": "gurobi"}
all_scenario_names = []
for i in range(nr_scenarios):
    all_scenario_names.append("s"+str(i)) 
ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
results = ef.solve_extensive_form()
end_time = time.time()

objval = ef.get_objective_value()
soln = ef.get_root_solution()
firstStageSol=[]
for (var_name, var_val) in soln.items():
    if var_val>0.5:
        index = var_name.split('[')[1].split(']')[0]
        firstStageSol.append(int(index)+1)
print(file_path,nr_scenarios)
print(firstStageSol,round(objval, 3),round(end_time - start_time,2))







