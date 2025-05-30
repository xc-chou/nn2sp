# Description:
# Train a neural network (nn) as a surrogate model for the recourse problem
# Solve the optimization problem using the embedded nn via gurobi machine learning

# Example:
# python surrogate.py ts-cap61-100-10000.txt Data/cap61 15

# Tunable Parameters:
# Network architecture, training set, 
# The last input value is to fix partially the solution based on the optimal deterministic solution 
# If set to 0, the entire solution is fixed as the deterministic optimal solution 

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from gurobi_ml import add_predictor_constr

# Read Files
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

    return num_locations, num_customers, locations

def read_training_set(training_set):
    X = []
    Y = []
    with open(training_set, 'r') as file:
        for line in file:
            parts = line.strip().split(";")
            numbers = parts[0]
            value = float(parts[1])
            X.append(eval(numbers))
            Y.append(value)
    return X, Y

def binary_sol(input):
    binary_sol = []
    for i in range(1, num_locations + 1):
        if i in input:
            binary_sol.append(1)
        else:
            binary_sol.append(0)
    return binary_sol

def find_max(X):
    max_num = float('-inf') 
    for sublist in X:
        if sublist:
            sublist_max = max(sublist)
            if sublist_max > max_num:
                max_num = sublist_max
    return max_num

training_set = sys.argv[1]
X_all,Y_all = read_training_set(training_set)
num_locations = find_max(X_all)
optX = X_all[0]
optY = Y_all[0]

X = X_all[1:]
Y = Y_all[1:]
num_samples = len(X)
X = list(map(binary_sol, X))

split_ratio = 0.8
split_index = int(num_samples * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# NN regression
# The network architecture is customizable 
layers = [50] * 5
regression = MLPRegressor(hidden_layer_sizes=layers, activation="relu",max_iter=1000)
pipe = make_pipeline(PolynomialFeatures(), regression)
pipe.fit(X=X_train, y=Y_train)

# Performance of the NN
r2_score = metrics.r2_score(Y_test, pipe.predict(X_test))
mape = np.mean(np.abs((Y_test - pipe.predict(X_test)) / Y_test)) * 100
check = abs(pipe.predict([binary_sol(optX)])-optY)/optY
print("R2 score {}, mape {} check error {}".format(r2_score, mape, check))
mape_train = np.mean(np.abs((Y_train - pipe.predict(X_train)) / Y_train)) * 100
print("train {} test {}".format(mape_train, mape))

# Optimization 
file_path = sys.argv[2]
num_locations, num_customers, locations = read_dataset(file_path)

model = gp.Model("SSCFLP")

x = model.addVars(num_locations, vtype=GRB.BINARY, name="x") 
y_approx = model.addVar(lb=-GRB.INFINITY, name="y")

model.setObjective(
    gp.quicksum(locations[i]['fixed_costs'] * x[i] for i in range(num_locations)) + y_approx,
    sense=GRB.MINIMIZE
)

pred_constr = add_predictor_constr(model, pipe, x, y_approx)
#pred_constr.print_stats()

# optional-check: fix/free a few variables 
minues = sys.argv[3]
minues = int(minues)
for i in range(num_locations-minues):
    model.addConstr(x[i]==binary_sol(optX)[i])

model.Params.TimeLimit = 1200
model.Params.MIPGap = 0.1
model.Params.NonConvex = 2

#model.setParam('OutputFlag', 0) 
model.optimize()

print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)

open_facilities=[]
for i in range(num_locations):
    if x[i].x > 0.5:
        open_facilities.append(i+1) 
print(open_facilities)

