import numpy as np
import time


def SEO(agents, objective_function, ub, lb, max_iter):
    num_agents, num_dimensions = agents.shape
    bounds = [ub, lb]
    ct = time.time()
    best_agent = None
    best_fitness = np.inf
    convergence = np.zeros(max_iter)
    for iteration in range(max_iter):
        for i in range(num_agents):
            agent = agents[i]
            fitness = objective_function(agent)

            if fitness < best_fitness:
                best_agent = agent.copy()
                best_fitness = fitness

            # Social interaction
            neighbors = [x for j, x in enumerate(agents) if j != i]
            random_neighbor = neighbors[np.random.randint(len(neighbors))]
            new_agent = agent + np.random.uniform(-1, 1, size=num_dimensions) * (random_neighbor - agent)

            # Ensure new agent stays within bounds
            new_agent = np.clip(new_agent, bounds[0], bounds[1])

            # Update agent if new fitness is better
            new_fitness = objective_function(new_agent)
            if new_fitness < fitness:
                agents[i] = new_agent

        convergence[iteration] = np.min(best_agent)
    ct = time.time() - ct
    return best_fitness, convergence, best_agent, ct



import numpy as np
import time

# def SEO(agents, objective_function, ub, lb, max_iter):
#     def spot_attack(defender):
#         return np.random.uniform(low=np.min(defender) - 0.1, high=np.max(defender) + 0.1, size=num_dimensions)
#
#     def check_boundary(agent, bounds):
#         return np.clip(agent, bounds[0], bounds[1])
#
#     def exchange_positions(attacker, defender, obj_func):
#         if obj_func(defender) < obj_func(attacker):
#             return defender
#         else:
#             return attacker
#
#     def generate_new_solution(defender, best, population, max_iter, num_Q, partition, BAR):
#         num_dimensions = len(defender)
#         num_agents = 2
#         max_step_size = 1
#         It = 1
#         Q = np.zeros((num_Q, num_dimensions))
#
#         while It <= max_iter:
#             # Training and retraining
#             Num_attack = 1
#             while Num_attack <= max_iter:
#                 # Spot an attack
#                 attack = spot_attack(defender)
#                 # Check the boundary
#                 attack = check_boundary(attack, bounds)
#                 # Respond to attack
#                 defender = exchange_positions(attacker, attack, objective_function)
#                 Num_attack += 1
#
#             for i in range(num_Q):
#                 defender = generate_new_solution(defender, attacker, attacker, max_iter, num_Q, partition, BAR)
#
#             It += 1
#
#         return defender
#
#     num_dimensions = len(ub)
#     num_agents = 2
#
#     start_time = time.time()
#
#     # Run the optimization
#     best_agent = generate_new_solution(agents[0], agents[1], agents[0], max_iter, len(agents), 0.5, 0.5)
#     best_fitness = objective_function(best_agent)
#
#     end_time = time.time()
#     ct = end_time - start_time
#
#     convergence = []  # Placeholder, convergence tracking not implemented
#
#     return best_fitness, convergence, best_agent, ct


# Example usage:
def sphere_function(x):
    return np.sum(x ** 2)


from numpy import matlib

if __name__ == '__main__':

    # bounds = (-5.12, 5.12)  # Bounds for each dimension
    # num_agents = 50
    # num_dimensions = 10
    # max_iter = 1000
    #
    # best_solution, best_fitness = seo(objective_function=sphere_function, num_agents=num_agents,
    #                                   num_dimensions=num_dimensions, bounds=bounds, max_iter=max_iter)

    Npop = 10
    Chlen = 3
    xmin = matlib.repmat([50, 0, 5], Npop, 1)
    xmax = matlib.repmat([100, 1, 15], Npop, 1)
    fname = sphere_function

    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 25

    print("DHOA...")
    [bestfit1, fitness1, bestsol1, time1] = SEO(initsol, fname, xmin, xmax, Max_iter)
    print("Best Solution:", bestsol1)
    print("Best Fitness:", bestfit1)
