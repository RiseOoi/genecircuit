import numpy as np
import multiprocessing
import sys
import time
import matplotlib.pyplot as plt

# =============================================================================
# Distributed Computing Parameters
pool_size = multiprocessing.cpu_count()

# Genetic Circuit Hyperparameters
NODES = 3000

# Evolutionary Algorithm Hyperparameters
GENERATIONS = 201  # number of generations to run

# Other Hyperparameters
# STEP_MUTATION_RATE = 0.9
# BIG_STEP_MUTATION_RATE = 0.8
# RANDOM_MUTATION_RATE = 1
# SIGN_FLIP_MUTATION_RATE = 0.1

# REG_RATE = 0.0003  # regularization rate
STEP_SIZE = 2.0  # max mutation intensity of each weight
POPULATION = pool_size * 6  # total number of population
SURVIVABLE_PARENTS = POPULATION // 3  # number of parents to survive

# Novelty Search Hyperparameters
# KNN_BC_NUM = 1  # k nearest neighbors number for behavior characteristics
# ARCHIVE_STORING_RATE = 0.01

# ODE
TIME_STEPS = 300
BATCH_SIZE = 30  # Fully dividable by 3 recommended

# Score Constraints
ERROR_BOUND = 0.1  # percentage of error allowed (sigmoid bounds are +-1)
BANDPASS_BOUND = 0.3

# the absolute bound of each weight (very important)
# choose something close to sigmoid saturation is good (eg. 7.5+, 5 is not good, 10 is good)
BOUND = 13

# Parameters (Derived from hyperparameters)
DNA_SIZE = NODES * NODES
UPPER_BANDPASS_BOUND = 1 - BANDPASS_BOUND
COST_UPPER_BOUND = ERROR_BOUND * BATCH_SIZE


# =============================================================================
# Mean normalization
def standardize(population):
    # as known as z-score normalization
    # the other method being min-max normalization
    for i, weights in enumerate(population):
        mean = np.mean(weights)
        std = np.std(weights)
        population[i] = (weights - mean) / std

    return population


# =============================================================================
# ODE & Simulations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# FF Classifier
# Here, only the classical solution determinator is implemented
# def simulate_ode_original(W, N, B, S):
#     dt = 0.01
#     initial_val = 0.1 * np.ones([B, S])  # can we reuse this?
#     input_val = np.linspace(0, 2, B).reshape(B, 1) * np.random.normal(
#         loc=1.0, scale=0.0001, size=[N, B, S])  # can we reduce the redundants?
#     input_val[:, :, 1:S] = 0.0

#     output = initial_val + (
#         sigmoid(np.matmul(initial_val, W)) - initial_val + input_val[0]) * dt

#     # print(output)
#     # HOW: create one time np.linspace(0, 2, B), mutate and reuse in for loop
#     for i in range(1, N):
#         output = output + (
#             sigmoid(np.matmul(output, W)) - output + input_val[i]) * dt

#     # print(output)

#     return output

# input_initializer = np.linspace(0, 2, BATCH_SIZE).reshape(BATCH_SIZE, 1,)
# input_val[:, 0] = np.linspace(0, 2, BATCH_SIZE).reshape(BATCH_SIZE)
# print(np.random.normal(loc=1.0, scale=0.0001))

dt = 0.01
initial_val = 0.1 * np.ones([BATCH_SIZE, NODES])
input_val = np.zeros((BATCH_SIZE, NODES))
linspace_col = np.linspace(0, 2, BATCH_SIZE).reshape(BATCH_SIZE)


def simulate_ode(W, N, B, S):
    # Insert one input and have three outputs
    input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
    input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
    input_val[:, 2] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, W)) - initial_val + input_val) * dt
    )

    for i in range(1, N):
        input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        input_val[:, 2] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = output + (sigmoid(np.matmul(output, W)) - output + input_val) * dt

    # print(output)
    return output


def plot_expressions(y, B):
    b = np.linspace(1, B, B)
    plt.title(f"{NODES} Nodes")

    plt.plot(b, y[:, 0], "black", linewidth=2, label="Input Node #1")
    plt.plot(b, y[:, 1], "saddlebrown", linewidth=2, label="Input Node #2")

    for i in range(3, y.shape[1] - 1):
        # plt.plot(b, y[:, i], 'g-', linewidth=2, label='Support Node')
        plt.plot(b, y[:, i], "gray", linewidth=2)

    plt.plot(b, y[:, -3], "b", linewidth=2, label="Output Node #3 - Switch")
    plt.plot(b, y[:, -2], "g", linewidth=2, label="Output Node #2 - Valley")
    plt.plot(b, y[:, -1], "r", linewidth=2, label="Output Node #1 - Bandpass")
    plt.xlabel("Input Level")
    plt.ylabel("Output Level")
    plt.legend()
    plt.show()


# =============================================================================
# Behavior characteristic distance mean calculator
# def population_novelty(population):
#     pop_novelty = np.zeros(POPULATION)
#     bc_distance = np.zeros(POPULATION)
#     for i, weights in enumerate(population):
#         for j, target in enumerate(population):
#                 bc_distance[j] = np.linalg.norm(weights - target)

#         # only uses KNN_BC_NUM of bc_distance to calculate bc_dist_mean
#         bc_distance.sort()
#         pop_novelty[i] = np.mean(bc_distance[-KNN_BC_NUM:])

#     return pop_novelty

# =============================================================================
# The forever (unforgettable) archive of most novel children in a generation
# Or another method: Prob 1% to store any children to archive
# archive = []

# =============================================================================
# Double mergesort sorting by alist
def double_mergesort(alist, blist):
    # print("Splitting ",alist)
    if len(alist) > 1:
        mid = len(alist) // 2
        lefthalf_a = alist[:mid]
        lefthalf_b = blist[:mid]
        righthalf_a = alist[mid:]
        righthalf_b = blist[mid:]

        double_mergesort(lefthalf_a, lefthalf_b)
        double_mergesort(righthalf_a, righthalf_b)

        i = 0
        j = 0
        k = 0
        while i < len(lefthalf_a) and j < len(righthalf_a):
            if lefthalf_a[i] < righthalf_a[j]:
                alist[k] = lefthalf_a[i]
                blist[k] = lefthalf_b[i]
                i = i + 1
            else:
                alist[k] = righthalf_a[j]
                blist[k] = righthalf_b[j]
                j = j + 1
            k = k + 1

        while i < len(lefthalf_a):
            alist[k] = lefthalf_a[i]
            blist[k] = lefthalf_b[i]
            i = i + 1
            k = k + 1

        while j < len(righthalf_a):
            alist[k] = righthalf_a[j]
            blist[k] = righthalf_b[j]
            j = j + 1
            k = k + 1


# =============================================================================
# Main functions
# Bandpass Determinator
# Determines whether the solution given is a bandpass
# so that you don't need the flags -> faster
def bandpass_determinator(y):
    # here we check only one node
    # it would be wise to check other nodes, to check if it is classical solution
    starting_low_flag = False
    middle_high_flag = False
    ending_low_flag = False
    for pt in y[:, -1]:
        if not starting_low_flag:
            if pt < BANDPASS_BOUND:
                starting_low_flag = True
        elif not middle_high_flag:
            if pt > UPPER_BANDPASS_BOUND:
                middle_high_flag = True
        elif not ending_low_flag:
            if pt < BANDPASS_BOUND:  # something is wrong here
                ending_low_flag = True
        else:
            if pt > BANDPASS_BOUND:
                ending_low_flag = False

    # print(starting_low_flag, middle_high_flag, ending_low_flag)

    return starting_low_flag and middle_high_flag and ending_low_flag


# Bandpass Cost function (for objective based selection method, the lower the better)
# Assume pt size is dividable by three
bandpass_design = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
bandpass_design = np.array(bandpass_design)


def bandpass_cost_calculator(y, B):
    cost = np.sum(np.abs(y - bandpass_design))

    return cost


def switch_cost_calculator(y, B):
    cost = 0
    for pt in y[: B // 2]:
        cost += np.absolute(pt - 0)
    for put in y[B // 2 :]:
        cost += np.absolute(1 - pt)

    return cost


def linear_cost_calculator(y, B):
    B -= 1
    cost = 0
    for i, pt in enumerate(y):
        cost += np.absolute(pt - (i / B))

    return cost


peak_design = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.0,
    0.875,
    0.75,
    0.625,
    0.5,
    0.375,
    0.25,
    0.125,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
peak_design = np.array(peak_design)


def peak_cost_calculator(y, B):
    # Experiment failed: Made a mountain instead, much easier than bandpass...
    cost = np.sum(np.abs(y - peak_design))

    return cost


cosine_design = [
    1.0,
    0.9766205557100867,
    0.907575419670957,
    0.7960930657056438,
    0.6473862847818277,
    0.46840844069979015,
    0.26752833852922075,
    0.05413890858541761,
    -0.16178199655276473,
    -0.37013815533991445,
    -0.5611870653623823,
    -0.7259954919231308,
    -0.8568571761675893,
    -0.9476531711828025,
    -0.9941379571543596,
    -0.9941379571543596,
    -0.9476531711828025,
    -0.8568571761675892,
    -0.7259954919231307,
    -0.5611870653623825,
    -0.37013815533991445,
    -0.16178199655276476,
    0.05413890858541758,
    0.267528338529221,
    0.4684084406997903,
    0.6473862847818279,
    0.796093065705644,
    0.9075754196709569,
    0.9766205557100867,
    1.0,
]
cosine_design = np.array(cosine_design)


def cosine_cost_calculator(y, B):
    cost = np.sum(np.abs(y - cosine_design))

    return cost


# valley_design = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9458172417006346, 0.7891405093963936, 0.546948158122427, 0.24548548714079924, -0.08257934547233227, -0.40169542465296926, -0.6772815716257409, -0.879473751206489, -0.9863613034027223, -0.9863613034027224, -0.8794737512064891, -0.6772815716257414, -0.40169542465296987, -0.08257934547233274, 0.2454854871407988, 0.5469481581224266, 0.7891405093963934, 0.9458172417006346, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# valley_design = 1 - bandpass_design
# valley_design = 1 - peak_design
def valley_cost_calculator(y, B):
    cost = np.sum(np.abs(y - valley_design))

    return cost


bandpass_reversed_design = 1 - bandpass_design


def bandpass_reversed_cost_calculator(y, B):
    cost = np.sum(np.abs(y - bandpass_reversed_design))

    return cost


# def adaptation_cost_calculator(y, B):
#     cost = 0
#     ADAPTED_LEVEL = 0.1
#     for pt in y[:B // 3]:
#         cost += np.absolute(pt - 0)
#     slice = ((1- ADAPTED_LEVEL) / (B//3))
#     for i, pt in enumerate(y[B // 3:2 * B // 3]):
#         cost += np.absolute(1 - i * slice) * 3
#         print(1 - i * slice)
#     sys.exit()
#     for pt in y[2 * B // 3:]:
#         cost += np.absolute(pt - ADAPTED_LEVEL)

#     return cost


adaptation_design = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
    0.00048828125,
    0.000244140625,
    0.0001220703125,
    6.103515625e-05,
    3.0517578125e-05,
    1.52587890625e-05,
    7.62939453125e-06,
    3.814697265625e-06,
    1.9073486328125e-06,
]
adaptation_design = np.array(adaptation_design)


def adaptation_cost_calculator(y, B):
    cost = 0
    # for i, pt in enumerate(y):
    #     cost += np.absolute(pt - adaptation_design[i])
    cost = np.sum(np.abs(y - adaptation_design))

    return cost


# # def adaptation_cost_calculator(y, B):
#     cost = 0
#     for pt in y[:B // 3]:
#         cost += np.absolute(pt - 0)
#     for pt in y[B // 3:2 * B // 3]:
#         cost += np.absolute(1 - pt)
#     for pt in y[2 * B // 3:]:
#         cost += np.absolute(pt - 0.5)

#     return cost


# Fitness based
cost_storage = [-1] * POPULATION
# def select(population):
#     for i, potential_parent in enumerate(population):
#         y = simulate_ode(potential_parent, TIME_STEPS, BATCH_SIZE, NODES)
#         # Multiple outputs
# cost_storage[i] = bandpass_cost_calculator(y[:, -1], BATCH_SIZE) * 1.5
#         cost_storage[i] += switch_cost_calculator(y[:, -2], BATCH_SIZE) * 1.25
#         # cost_storage[i] = adaptation_cost_calculator(y[:, -1], BATCH_SIZE)
#         cost_storage[i] += linear_cost_calculator(y[:, -3], BATCH_SIZE)
#         cost_storage[i] /= 3
#         # cost_storage[i] += REG_RATE * sum(sum(abs(potential_parent)))  # regularization

#     double_mergesort(cost_storage, population)

#     y = simulate_ode(population[0], TIME_STEPS, BATCH_SIZE, NODES)
#     print("Bandpass Cost:", bandpass_cost_calculator(y[:, -1], BATCH_SIZE))
#     print("Switch Cost:", switch_cost_calculator(y[:, -2], BATCH_SIZE))
#     print("Linear Cost:", linear_cost_calculator(y[:, -3], BATCH_SIZE))
#     # print(cost_storage[0])
#     survivors = population[:SURVIVABLE_PARENTS]
#     survivors = np.append(survivors, survivors, axis=0)
#     # repopulated_parents = np.append(repopulated_parents, survivors, axis=0)
#     # random_children = np.random.uniform(-BOUND, BOUND, (SURVIVABLE_PARENTS, NODES, NODES))
#     # survivors = np.append(repopulated_parents, random_children, axis=0)
#     # print(repopulated_parents)
#     return survivors, population[0], cost_storage[0]

# def select(population):
#     # Harmonic Version - Mitigate Impact of Outliers
#     for i, potential_parent in enumerate(population):
#         y = simulate_ode(potential_parent, TIME_STEPS, BATCH_SIZE, NODES)
#         # Multiple outputs
#         f_bandpass = BATCH_SIZE - bandpass_cost_calculator(y[:, -1], BATCH_SIZE)
#         f_switch = BATCH_SIZE - switch_cost_calculator(y[:, -2], BATCH_SIZE)
#         f_linear = BATCH_SIZE - linear_cost_calculator(y[:, -3], BATCH_SIZE)
#         cost_storage[i] = BATCH_SIZE - 3 / (((1/f_bandpass) + (1/f_switch) + (1/f_linear)))
#         # cost_storage[i] += REG_RATE * sum(sum(abs(potential_parent)))  # regularization
#         # cost_storage[i] = f_bandpass + f_switch + f_linear

#     double_mergesort(cost_storage, population)

#     y = simulate_ode(population[0], TIME_STEPS, BATCH_SIZE, NODES)
#     print("Bandpass Cost:", bandpass_cost_calculator(y[:, -1], BATCH_SIZE))
#     print("Switch Cost:", switch_cost_calculator(y[:, -2], BATCH_SIZE))
#     print("Linear Cost:", linear_cost_calculator(y[:, -3], BATCH_SIZE))
#     # print(cost_storage[0])
#     survivors = population[:SURVIVABLE_PARENTS]
#     survivors = np.append(survivors, survivors, axis=0)
#     # repopulated_parents = np.append(repopulated_parents, survivors, axis=0)
#     # random_children = np.random.uniform(-BOUND, BOUND, (SURVIVABLE_PARENTS, NODES, NODES))
#     # survivors = np.append(repopulated_parents, random_children, axis=0)
#     # print(repopulated_parents)
#     return survivors, population[0], cost_storage[0]

# def select(population):
#     # Square Version - Aggravate Impact of Outliers
#     for i, potential_parent in enumerate(population):
#         y = simulate_ode(potential_parent, TIME_STEPS, BATCH_SIZE, NODES)
#         # Multiple outputs
#         f_bandpass = bandpass_cost_calculator(y[:, -1], BATCH_SIZE)
#         f_bandpass_reversed = bandpass_reversed_cost_calculator(y[:, -2], BATCH_SIZE)
#         f_switch = switch_cost_calculator(y[:, -3], BATCH_SIZE)
#         # f_valley = valley_cost_calculator(y[:, -3], BATCH_SIZE)
#         # f_linear = linear_cost_calculator(y[:, -3], BATCH_SIZE)
#         # cost_storage[i] = valley_cost_calculator(y[:, -1], BATCH_SIZE)
#         # cost_storage[i] = peak_cost_calculator(y[:, -1], BATCH_SIZE)
#         # cost_storage[i] = bandpass_cost_calculator(y[:, -1], BATCH_SIZE)
#         cost_storage[i] = f_bandpass**2 + f_switch**2 + f_bandpass_reversed**2
#         # cost_storage[i] += REG_RATE * sum(sum(abs(potential_parent)))  # regularization
#         # cost_storage[i] = f_bandpass + f_switch + f_linear

#     double_mergesort(cost_storage, population)

#     y = simulate_ode(population[0], TIME_STEPS, BATCH_SIZE, NODES)
#     print("Bandpass Cost:", bandpass_cost_calculator(y[:, -1], BATCH_SIZE))
#     print("Valley Cost:", bandpass_reversed_cost_calculator(y[:, -2], BATCH_SIZE))
#     print("Switch Cost:", switch_cost_calculator(y[:, -3], BATCH_SIZE))
#     # print("Valley Cost:", valley_cost_calculator(y[:, -3], BATCH_SIZE))
#     # print("Linear Cost:", linear_cost_calculator(y[:, -3], BATCH_SIZE))
#     # print(cost_storage[0])
#     survivors = population[:SURVIVABLE_PARENTS]
#     survivors = np.append(survivors, survivors, axis=0)
#     # repopulated_parents = np.append(repopulated_parents, survivors, axis=0)
#     # random_children = np.random.uniform(-BOUND, BOUND, (SURVIVABLE_PARENTS, NODES, NODES))
#     # survivors = np.append(repopulated_parents, random_children, axis=0)
#     # print(repopulated_parents)
#     return survivors, population[0], cost_storage[0]


def select(population):
    for i, potential_parent in enumerate(population):
        f_bandpass = simulate_and_cost_bandpass(potential_parent)
        f_bandpass_reversed = simulate_and_cost_bandpass_reversed(potential_parent)
        f_switch = simulate_and_cost_switch(potential_parent)
        cost_storage[i] = f_bandpass ** 2 + f_bandpass_reversed ** 2 + f_switch ** 2

    double_mergesort(cost_storage, population)

    survivors = population[:SURVIVABLE_PARENTS]
    survivors = np.append(survivors, survivors, axis=0)

    return survivors, population[0], cost_storage[0]


def plot(y):
    b = np.linspace(1, BATCH_SIZE, BATCH_SIZE)
    plt.title(f"{NODES} Nodes")

    plt.plot(b, y[:, 0], "black", linewidth=2, label="Input Node #1")
    plt.plot(b, y[:, 1], "saddlebrown", linewidth=2, label="Input Node #2")

    for i in range(2, y.shape[1] - 1):
        # plt.plot(b, y[:, i], 'g-', linewidth=2, label='Support Node')
        plt.plot(b, y[:, i], "gray", linewidth=2)

    plt.plot(b, y[:, -1], "r", linewidth=2, label="Multifunction Output Node")
    plt.xlabel("Input Level")
    plt.ylabel("Output Level")
    plt.legend()
    plt.show()


def simulate_and_cost_bandpass(individual):
    # Encode <- 0, 1
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    cost = np.sum(np.abs(output[:, -1] - bandpass_design))

    return cost


def simulate_and_cost_bandpass_reversed(individual):
    # Encode  <- 1, 0
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    cost = np.sum(np.abs(output[:, -1] - bandpass_reversed_design))

    return cost


switch_design = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
switch_design = np.array(switch_design)


def simulate_and_cost_switch(individual):
    # Encode <- 1, 1
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
    input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    cost = np.sum(np.abs(output[:, -1] - switch_design))

    return cost


def simulate_plot_cost_bandpass(individual):
    # Encode <- 0, 1
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    plot(output)


def simulate_and_plot_bandpass_reversed(individual):
    # Encode  <- 1, 0
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    plot(output)


def simulate_and_plot_switch(individual):
    # Encode <- 1, 1
    input_val = np.zeros((BATCH_SIZE, NODES))
    input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
    input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)

    output = (
        initial_val
        + (sigmoid(np.matmul(initial_val, individual)) - initial_val + input_val) * dt
    )

    for i in range(1, TIME_STEPS):
        input_val[:, 0] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        input_val[:, 1] = linspace_col * np.random.normal(loc=1.0, scale=0.0001)
        output = (
            output + (sigmoid(np.matmul(output, individual)) - output + input_val) * dt
        )

    plot(output)


def distributed_select(population):
    pass


# Mutation
def mutate(population):
    # doesn't mutate the elite
    for p in range(1, len(population)):
        for i in range(NODES):
            for j in range(NODES):
                if np.random.rand() < RANDOM_MUTATION_RATE:
                    population[p][i][j] = (
                        BOUND * np.random.rand() * (-1) ** np.random.randint(2)
                    )
                elif np.random.rand() < SIGN_FLIP_MUTATION_RATE:
                    population[p][i][j] = -1 * population[p][i][j]
                else:
                    population[p][i][j] += (
                        STEP_SIZE * np.random.rand() * (-1) ** np.random.randint(2)
                    )
                # population[p][i][j] += 100

    # print(population)

    return population


def original_mutate(population):
    for p in range(1, len(population)):
        for i in range(NODES):
            for j in range(NODES):
                population[p][i][j] += (
                    STEP_SIZE * np.random.rand() * (-1) ** np.random.randint(2)
                )

    return population


def distributed_mutation(individual):
    for i in range(NODES):
        for j in range(NODES):
            individual[i][j] += (
                STEP_SIZE * np.random.rand() * (-1) ** np.random.randint(2)
            )

    return individual


def distributed_small_mutation(individual):
    for i in range(NODES):
        for j in range(NODES):
            if np.random.rand() < STEP_MUTATION_RATE:
                individual[i][j] += (
                    STEP_SIZE * np.random.rand() * (-1) ** np.random.randint(2)
                )
            else:
                individual[i][j] = (
                    BOUND * np.random.rand() * (-1) ** np.random.randint(2)
                )
            # elif np.random.rand() < SIGN_FLIP_MUTATION_RATE:
            #     individual[i][j] = -1 * individual[i][j]

            # population[p][i][j] += 100

    return individual


def distributed_big_mutation(individual):
    for i in range(NODES):
        for j in range(NODES):
            if np.random.rand() < BIG_STEP_MUTATION_RATE:
                individual[i][j] += (
                    BIG_STEP_SIZE * np.random.rand() * (-1) ** np.random.randint(2)
                )
            else:
                individual[i][j] = (
                    BOUND * np.random.rand() * (-1) ** np.random.randint(2)
                )
            # elif np.random.rand() < SIGN_FLIP_MUTATION_RATE:
            #     individual[i][j] = -1 * individual[i][j]

            # population[p][i][j] += 100

    return individual


# =============================================================================
# Random Initialization Phase
population = np.random.uniform(-BOUND, BOUND, (POPULATION, NODES, NODES))
# print(population)
# population = standardize(population)
# print(population)

# multiprocessing pool initializer
pool = multiprocessing.Pool(pool_size)
# best_score = BATCH_SIZE
# best_elite = -1

# Genetic Algorithm Loop
for g in range(GENERATIONS):
    # Simulated Annealing
    # if g % 10 == 0 and STEP_SIZE > 0.1:
    STEP_SIZE -= 0.005
    BOUND -= 0.01
    # for g in range(1):
    # print(population)
    print("Generation:", g)
    start = time.time()
    survivors, elite, elite_score = select(population)
    end = time.time()
    print("Selection Time:", end - start)

    print("Elite Score:", elite_score)

    # if g % 10 == 0:
    np.save(
        f"large_controllability_generation_result_3000/controllability-encoded-2-in-1-out-generation-{g}.npy",
        elite,
    )

    # if elite_score < best_score:
    #     best_score = elite_score
    #     best_elite = survivors[0]

    # print("Elite:\n", elite)
    # print("10th:\n", population[9])
    # break if found t he solution
    # print(COST_UPPER_BOUND)
    # if elite_score < COST_UPPER_BOUND:
    # if elite_score < 27:
    #     break

    # population = crossover(population)
    start = time.time()
    survivors = np.array(pool.map(distributed_mutation, survivors))
    # survivors = original_mutate(survivors)
    # print(survivors[0])
    # print(elite)
    survivors[0] = elite
    population = np.append(
        survivors,
        np.random.uniform(-BOUND, BOUND, (SURVIVABLE_PARENTS, NODES, NODES)),
        axis=0,
    )
    # population = original_mutate(population)
    # elite = population[0]
    # first_population = np.array(pool.map(distributed_small_mutation, population[:SURVIVABLE_PARENTS]))
    # second_population = np.array(pool.map(distributed_big_mutation, population[SURVIVABLE_PARENTS:SURVIVABLE_PARENTS*2]))
    # population = np.append(first_population, population[SURVIVABLE_PARENTS*2:], axis=0)
    # population = np.append(population, second_population, axis=0)
    # population = np.append([elite], end_population, axis=0)
    end = time.time()
    # print(population[0])
    # print(type(population))
    # print(population.shape)
    print("Multiprocessing Mutation Time:", end - start)

    # print(sum(sum(population[0] - population[SURVIVABLE_PARENTS])))
    # print(population)
    # population = standardize(population)
    # print(population)
    print()

# experimental results
# np.save(f'{NODES}-nodes-multifunctions.npy', population[0])    # .npy extension is added if not given
# np.save(f'multifunctions-2_encoded_input-1-output.npy', elite)    # .npy extension is added if not given
# d = np.load('test3.npy')

simulate_plot_cost_bandpass(elite)
simulate_and_plot_bandpass_reversed(elite)
simulate_and_plot_switch(elite)

# y = simulate_ode(population[1], TIME_STEPS, BATCH_SIZE, NODES)
# plot_expressions(y, BATCH_SIZE)
# y = simulate_ode(population[2], TIME_STEPS, BATCH_SIZE, NODES)
# plot_expressions(y, BATCH_SIZE)
# y = simulate_ode(population[3], TIME_STEPS, BATCH_SIZE, NODES)
# plot_expressions(y, BATCH_SIZE)
# y = simulate_ode(population[4], TIME_STEPS, BATCH_SIZE, NODES)
# plot_expressions(y, BATCH_SIZE)