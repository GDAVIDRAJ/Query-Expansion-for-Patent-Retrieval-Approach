import numpy as np
import time


def PROPOSED(agents, objective_function, ub, lb, max_iter):
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

