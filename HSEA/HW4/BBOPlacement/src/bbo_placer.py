import os
import random
import argparse
import csv
import datetime
import numpy as np
import yaml

from place_db import PlaceDB
from utils import write_final_placement, rank_macros, evaluate

from utils import initialize_population, selection, mutation, crossover


class BBOPlacer:
    """Basic class for WireMask-BBO"""
    def __init__(self, dim, grid_num, grid_size, placedb, node_id_ls, csv_writer, csv_file, placement_save_dir):
        self.dim = dim
        self.lb = 0 * np.ones(dim)
        self.ub = grid_num * np.ones(dim)
        self.grid_num = grid_num
        self.grid_size = grid_size
        self.placedb = placedb
        self.node_id_ls = node_id_ls
        self.csv_writer = csv_writer
        self.csv_file = csv_file
        self.best_hpwl = 1e12
        self.placement_save_dir = placement_save_dir

    def _evaluate(self, x):
        """
        Evaluate by WireMask-BBO

        Returns:
            hpwl value of solution x
        """
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        place_record = {}
        node_id_ls = self.node_id_ls.copy()
        for i in range(len(node_id_ls)):
            place_record[node_id_ls[i]] = {}
            place_record[node_id_ls[i]]["loc_x"] = x[i * 2]
            place_record[node_id_ls[i]]["loc_y"] = x[i * 2 + 1]
        placed_macro, hpwl = evaluate(self.node_id_ls, self.placedb, self.grid_num, self.grid_size, place_record)
        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
            write_final_placement(placed_macro, self.placement_save_dir)
        self.csv_writer.writerow([hpwl])
        self.csv_file.flush()
        return hpwl


class RandomSearch:
    """Simple implementation of Random search."""

    def __init__(self, placer, max_iteration: int = 1000):
        """
        Initialize the RandomSearch object.

        Args:
            placer: An object representing the macro placer.
            max_iteration: Maximum number of iterations (default: 1000).
        """
        self.placer = placer
        self.max_iteration = max_iteration

    def init(self):
        """Initialize the search by generating a random solution."""
        self.x = np.random.randint(self.placer.lb, self.placer.ub + 1, self.placer.dim)

    def step(self):
        """Take a step in the search by generating a new random solution. RS just randomly generate a new solution"""
        self.init()

    def evaluate(self):
        """Evaluate the current solution."""
        return self.placer._evaluate(self.x)

    def run(self):
        """Run the random search algorithm."""
        self.init()
        for i in range(self.max_iteration):
            value = self.evaluate()
            print(f'HPWL at iteration {i} is {value}')
            self.step()
            
class SimulatedAnnealingRandomSearch:
    """Implementation of Random search with simulated annealing."""

    def __init__(self, placer, max_iteration: int = 1000, initial_temperature: float = 100.0, cooling_rate: float = 0.95):
        """
        Initialize the SimulatedAnnealingRandomSearch object.

        Args:
            placer: An object representing the macro placer.
            max_iteration: Maximum number of iterations (default: 1000).
            initial_temperature: Initial temperature for simulated annealing (default: 100.0).
            cooling_rate: Cooling rate for simulated annealing (default: 0.95).
        """
        self.placer = placer
        self.max_iteration = max_iteration
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def init(self):
        """Initialize the search by generating a random solution."""
        self.x = np.random.randint(self.placer.lb, self.placer.ub + 1, self.placer.dim)

    def step(self, temperature):
        """Take a step in the search by generating a new random solution with simulated annealing."""
        x_new = self.x + np.asarray([random.randint(-1, 1) for _ in range(self.placer.dim)])  # Generate a new solution by randomly perturbing the current solution
        x_new = np.clip(x_new, self.placer.lb, self.placer.ub)  # Clip the new solution to the feasible range
        delta_cost = self.placer._evaluate(x_new) - self.placer._evaluate(self.x)  # Calculate the cost difference between the new and current solution
        
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / temperature):
            # Accept the new solution if it has a lower cost or with a probability based on the temperature
            self.x = x_new

    def evaluate(self):
        """Evaluate the current solution."""
        return self.placer._evaluate(self.x)

    def run(self):
        """Run the simulated annealing random search algorithm."""
        self.init()
        temperature = self.initial_temperature
        for i in range(self.max_iteration):
            value = self.evaluate()
            print(f'HPWL at iteration {i} is {value}')
            self.step(temperature)
            temperature *= self.cooling_rate  # Reduce the temperature at each iteration
            
# def get_dimension(ls):
#     if isinstance(ls, list):
#         return 1 + max(get_dimension(item) for item in ls)
#     else:
#         return 0

def main(args):
    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    dataset = args.dataset
    # random.seed(args.seed)
    np.random.seed(args.seed)
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    method = config['method']

    # read data_set
    placedb = PlaceDB(dataset)
    node_id_ls = rank_macros(placedb, rank_key='area')
    # print("node_id_ls is:", node_id_ls)
    grid_setting = config['grid_setting']
    grid_num = grid_setting[dataset]["grid_num"]
    grid_size = grid_setting[dataset]["grid_size"]
    macro_num = len(placedb.node_info.keys())
    dim = 2 * macro_num
    # print("dim is:", dim)

    # save data directory
    hpwl_save_dir = f"./result/{method}/curve/"
    placement_result_save_dir = f"./result/{method}/placement_result/"
    if not os.path.exists(hpwl_save_dir):
        os.makedirs(hpwl_save_dir)
    if not os.path.exists(placement_result_save_dir):
        os.makedirs(placement_result_save_dir)
    if args.timestamp:
        hpwl_save_dir += f"{dataset}_{args.seed}_{timestamp}.csv"
        placement_result_save_dir += f"{dataset}_{args.seed}_{timestamp}.csv"
    else:
        hpwl_save_dir += f"{dataset}_{args.seed}.csv"
        placement_result_save_dir += f"{dataset}_{args.seed}.csv"
    hpwl_save_file = open(hpwl_save_dir, "a+")
    hpwl_writer = csv.writer(hpwl_save_file)
    print(f'Running {method} on {args.dataset} with seed {args.seed}')
    print(f'HPWL log is {hpwl_save_dir}')

    # Run
    # origin algorithm
    bbo_placer = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
                           node_id_ls=node_id_ls, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
                           placement_save_dir=placement_result_save_dir)
    
    # evolutionary algorithm 
    # population_size = 50
    # num_generations = 10
    # population = initialize_population(population_size, node_id_ls)
    # best_fitness = float('inf')
    # best_individual = None
    # for generation in range(num_generations):
    #     print("The {}th Generation is Starting.".format(generation))
    #     fitness_values = []
    #     for i in range(len(population)):
    #         bbo_placer = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=population[i], csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x = np.random.randint(bbo_placer.lb, bbo_placer.ub + 1, bbo_placer.dim)
    #         hpwl = bbo_placer._evaluate(x)
    #         fitness_values.append(hpwl)
            
    #     min_fitness = min(fitness_values)
    #     min_index = fitness_values.index(min_fitness)
    #     if min_fitness < best_fitness:
    #         best_fitness = min_fitness
    #         best_individual = population[min_index]
    #     new_population = []
    #     selected_population = selection(population, fitness_values)
    #     # sel_dim = get_dimension(selected_population)
    #     # print("sel_dim is:", sel_dim)
    #     for i in range(0, len(selected_population)-1, 2):
    #         while 1:
    #             idx_parent1 = random.randint(0, len(selected_population)-1)
    #             idx_parent2 = random.randint(0, len(selected_population)-1)
    #             if idx_parent1 != idx_parent2:
    #                 break
    #         parent1, parent2 = selected_population[idx_parent1], selected_population[idx_parent2]
    #         # print("parent1 is:", parent1)
    #         children1, children2 = crossover(parent1, parent2)
    #         mutation(children1)
    #         mutation(children2)
    #         children_ls = []
    #         hpwl_ls = []
    #         bbo_placer_p1 = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=parent1, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x_p1 = np.random.randint(bbo_placer_p1.lb, bbo_placer_p1.ub + 1, bbo_placer_p1.dim)
    #         hpwl_p1 = bbo_placer_p1._evaluate(x_p1)
    #         children_ls.append(parent1)
    #         hpwl_ls.append(hpwl_p1)
    #         bbo_placer_p2 = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=parent2, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x_p2 = np.random.randint(bbo_placer_p2.lb, bbo_placer_p2.ub + 1, bbo_placer_p2.dim)
    #         hpwl_p2 = bbo_placer_p2._evaluate(x_p2)
    #         children_ls.append(parent2)
    #         hpwl_ls.append(hpwl_p2)
    #         bbo_placer_c1 = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=children1, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x_c1 = np.random.randint(bbo_placer_c1.lb, bbo_placer_c1.ub + 1, bbo_placer_c1.dim)
    #         hpwl_c1 = bbo_placer_c1._evaluate(x_c1)
    #         children_ls.append(children1)
    #         hpwl_ls.append(hpwl_c1)
    #         bbo_placer_c2 = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=children2, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x_c2 = np.random.randint(bbo_placer_c2.lb, bbo_placer_c2.ub + 1, bbo_placer_c2.dim)
    #         hpwl_c2 = bbo_placer_c2._evaluate(x_c2)
    #         children_ls.append(children2)
    #         hpwl_ls.append(hpwl_c2)
    #         children_data = list(zip(hpwl_ls, children_ls))
    #         sorted_children_data = sorted(children_data, key=lambda x: x[0])
    #         min_children = [data[1] for data in sorted_children_data[:2]]
    #         for j in min_children:
    #             new_population.append(j)
            
    #     next_population = population + new_population
    #     offspring = []
    #     next_scores = []
    #     for i in range(len(next_population)):
    #         bbo_placer = bbo_placer = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=next_population[i], csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir)
    #         x = np.random.randint(bbo_placer.lb, bbo_placer.ub + 1, bbo_placer.dim)
    #         hpwl = bbo_placer._evaluate(x)
    #         next_scores.append([hpwl, next_population[i]])
    #     tmp_next_scores = sorted(next_scores, key=lambda Score: Score[0])
    #     for i in range(len(population)):
    #         offspring.append(tmp_next_scores[i][1])
    #     population = offspring
    # bbo_placer = BBOPlacer(dim=dim, grid_num=grid_num, grid_size=grid_size, placedb=placedb,
    #                        node_id_ls=best_individual, csv_writer=hpwl_writer, csv_file=hpwl_save_file,
    #                        placement_save_dir=placement_result_save_dir) 
    
    random_search = RandomSearch(placer=bbo_placer, max_iteration=args.max_iteration)
    random_search.run()
    
    # sa_random_search = SimulatedAnnealingRandomSearch(placer=bbo_placer, max_iteration=args.max_iteration)
    # sa_random_search.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bigblue1')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--max_iteration', type=int, default=200)
    parser.add_argument('--timestamp', action='store_true', help='If use the timestamp in name')
    args = parser.parse_known_args()[0]
    main(args=args)
