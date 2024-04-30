import networkx as nx
import numpy as np
import argparse
import matplotlib.pyplot as plt

from scipy.special import zeta
import time


def generate_regular_graph(args):
    # 这里简单以正则图为例, 鼓励同学们尝试在其他类型的图(具体可查看如下的nx文档)上测试算法性能
    # nx文档 https://networkx.org/documentation/stable/reference/generators.html
    graph = nx.random_graphs.random_regular_graph(d=args.n_d, n=args.n_nodes, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)


def generate_gset_graph(args):
    # 这里提供了比较流行的图集合: Gset, 用于进行分割
    dir = './Gset/'
    fname = dir + 'G' + str(args.gset_id) + '.txt'
    graph_file = open(fname)
    n_nodes, n_e = graph_file.readline().rstrip().split(' ')
    print(n_nodes, n_e)
    nodes = [i for i in range(int(n_nodes))]
    edges = []
    for line in graph_file:
        n1, n2, w = line.split(' ')
        edges.append((int(n1) - 1, int(n2) - 1, int(w)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph, len(graph.nodes), len(graph.edges)


def graph_generator(args):
    if args.graph_type == 'regular':
        return generate_regular_graph(args)
    elif args.graph_type == 'gset':
        return generate_gset_graph(args)
    else:
        raise NotImplementedError(f'Wrong graph_tpye')


# 在原始的框架代码中每次的划分方式为：将满足一定条件的节点和不满足该条件的结点分开
# 但每次x的选取是随机的，因此x中哪些元素满足该条件也是随机的，即每次结点的划分也是随机的
def get_fitness(graph, x, n_edges, threshold=0):
    # 这里我将-1修改为了0
    x_eval = np.where(x >= threshold, 1, -1)
    # print("x_eval ", x_eval)
    # 获得Cuts值需要将图分为两部分, 这里默认以0为阈值把解分成两块.
    # g1和g2中的元素是结点的下标
    g1 = np.where(x_eval == -1)[0]
    # print("g1 ",g1)
    g2 = np.where(x_eval == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges  # cut_size返回的是连接图的两部分g1,g2的桥的权重之和


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='regular')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=50)  # 可见一共有2的50000次方个解（默认情况下）
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=10)
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=500)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=50)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator', default=.3)
    args = parser.parse_known_args()[0]
    return args


# 生成种群
def generate_population(n_nodes, n_population):
    tmp_population = [np.random.uniform(-1, 1, n_nodes).tolist() for _ in range(n_population)]
    return tmp_population


# 将实值表示的个体转换为01串表示的个体
def tmp_population2population(tmp_population):
    population = []
    for i in range(len(tmp_population)):
        tmp = []
        for j in range(len(tmp_population[i])):
            if tmp_population[i][j] >= 0:
                tmp.append(1)
            else:
                tmp.append(0)
        population.append(tmp)
    return population


def score(individual, graph, n_edges):
    ind = np.array(individual)
    g1 = np.where(ind == 0)[0]
    # print(g1)
    g2 = np.where(ind == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges


# 自然选择
def selection(population, k, graph, n_edges, scores):
    # scores = [score(i, graph, n_edges) for i in population]
    idx = np.random.randint(len(population))
    for i in np.random.randint(0, len(population), k - 1):
        if scores[i] > scores[idx]:
            idx = i
    return population[idx]


# 选出父代
def generate_parents(population, k, graph, n_edges, scores):
    return [selection(population, k, graph, n_edges, scores) for _ in range(len(population) // 5)]


# 交叉

def crossover(parent1, parent2):
    children1, children2 = parent1.copy(), parent2.copy()
    crossover_position = np.random.randint(1, len(children1) - 1)
    # crossoverPos2 = np.random.randint(1, len(children1) - 1)
    children1 = parent1[:crossover_position] + parent2[crossover_position:]
    children2 = parent2[:crossover_position] + parent1[crossover_position:]
    return children1, children2

"""
def crossover(parent1, parent2):
    children1, children2 = parent1.copy(), parent2.copy()
    crossover_position1 = np.random.randint(1, len(children1) - 1)
    crossover_position2 = np.random.randint(crossover_position1, len(children1) - 1)
    children1 = parent1[:crossover_position1] + parent2[crossover_position1:crossover_position2] + parent1[crossover_position2:]
    children2 = parent2[:crossover_position1] + parent1[crossover_position1:crossover_position2] + parent2[crossover_position2:]
    return children1, children2
"""


# 变异
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]

def heavytail_mutation(individual, mutation_rate):
    n = len(individual)
    alpha = 0
    distribution = []
    linelist = np.arange(1.5, n, 0.5)
    #print('linelist', linelist)
    # define The discrete power-law distribution
    for i in range(len(linelist)):
        beta = linelist[i]
        beta_res = 0
        left_bound = zeta(beta) - (beta / (beta - 1)) * (n/2 ** (-beta + 1))
        right_bound = zeta(beta)
        for j in range(1, int(n/2)+1):
            C = 0
            C = C + j ** (-beta)
        # print('C', C)
        if left_bound <= C <= right_bound:
            beta_res = beta
            break
    # print('beta_res', beta_res)
    for x in range(1, int(n/2)+1):
        C = 0
        for j in range(1, int(n/2)+1):
            C = C + j ** (-beta_res)
        distribution_res = C ** (-1) + x ** (-beta_res)
        distribution.append(distribution_res)
    distribution = distribution / np.sum(distribution)
    # print('distribution', distribution, len(distribution), np.sum(distribution))
    alpha = np.random.choice([i for i in range(1, int(n/2)+1)], p=distribution)
    for i in range(n):
        if np.random.rand() * n < alpha:
            if individual[i] == 1:
                individual[i] = 0
            else:
                individual[i] = 1



# Hamming Distance
def distance(individual1, individual2):
    result = 0
    for i in range(len(individual1)):
        result += abs(individual1[i]-individual2[i])
    return result




# 产生子代
def children(population, mutation_rate, k, graph, n_edges, scores):
    Children = []
    parents = generate_parents(population, k, graph, n_edges, scores)

    for i in range(0, len(parents)-1, 2):
        while 1:
            idx_parent1 = np.random.randint(len(parents))
            idx_parent2 = np.random.randint(len(parents))
            if idx_parent1 != idx_parent2:
                break
        parent1, parent2 = parents[idx_parent1], parents[idx_parent2]
        children1, children2 = crossover(parent1, parent2)
        mutation(children1, mutation_rate)
        mutation(children2, mutation_rate)

        if distance(parent1, children1) + distance(parent2, children2) > distance(parent1, children2) + distance(parent2, children1):
            if score(parent1, graph, n_edges) > score(children2, graph, n_edges):
                children2 = parent1
            if score(parent2, graph, n_edges) > score(children1, graph, n_edges):
                children1 = parent2
        else:
            if score(parent1, graph, n_edges) > score(children1, graph, n_edges):
                children1 = parent1
            if score(parent2, graph, n_edges) > score(children2, graph, n_edges):
                children2 = parent2
        Children.append(children1)
        Children.append(children2)

    next_population = population + Children
    offspring = []
    next_scores = [[score(i, graph, n_edges), i] for i in next_population]
    tmp_next_scores = sorted(next_scores, key=lambda Score: Score[0], reverse=True)
    for i in range(len(population)):
        offspring.append(tmp_next_scores[i][1])
    return offspring

def heavytail_children(population, mutation_rate, k, graph, n_edges, scores):
    Children = []
    parents = generate_parents(population, k, graph, n_edges, scores)

    for i in range(0, len(parents)-1, 2):
        while 1:
            idx_parent1 = np.random.randint(len(parents))
            idx_parent2 = np.random.randint(len(parents))
            if idx_parent1 != idx_parent2:
                break
        parent1, parent2 = parents[idx_parent1], parents[idx_parent2]
        children1, children2 = crossover(parent1, parent2)

        heavytail_mutation(children1, mutation_rate)
        heavytail_mutation(children2, mutation_rate)

        if distance(parent1, children1) + distance(parent2, children2) > distance(parent1, children2) + distance(parent2, children1):
            if score(parent1, graph, n_edges) > score(children2, graph, n_edges):
                children2 = parent1
            if score(parent2, graph, n_edges) > score(children1, graph, n_edges):
                children1 = parent2
        else:
            if score(parent1, graph, n_edges) > score(children1, graph, n_edges):
                children1 = parent1
            if score(parent2, graph, n_edges) > score(children2, graph, n_edges):
                children2 = parent2
        Children.append(children1)
        Children.append(children2)

    next_populaton = population + Children
    offspring = []
    next_scores = [[score(i, graph, n_edges), i] for i in next_populaton]
    tmp_next_scores = sorted(next_scores, key=lambda Score: Score[0], reverse=True)
    for i in range(len(population)):
        offspring.append(tmp_next_scores[i][1])
    return offspring




def get_best_fitness(population, graph, n_edges):
    best_fitness = -1
    for i in range(len(population)):
        if score(population[i], graph, n_edges) > best_fitness:
            best_fitness = score(population[i], graph, n_edges)
    return best_fitness


# 演化
def evolutionary_algorithm(n_nodes, n_population, mutation_rate, times, k, graph, n_edges):
    tmp_population = generate_population(n_nodes, n_population)
    xx2 = []
    yy2 = []
    population = tmp_population2population(tmp_population)
    for generation in range(times):
        scores = [score(i, graph, n_edges) for i in population]
        Children = children(population, mutation_rate, k, graph, n_edges, scores)
        population = Children
        xx2.append(generation)
        yy2.append(get_best_fitness(population, graph, n_edges))
        print(generation, get_best_fitness(population, graph, n_edges))
    return xx2, yy2

def heavytail_evolutionary_algorithm(n_nodes, n_population, mutation_rate, times, k, graph, n_edges):
    tmp_population = generate_population(n_nodes, n_population)
    xx3 = []
    yy3 = []
    population = tmp_population2population(tmp_population)
    for generation in range(times):
        scores = [score(i, graph, n_edges) for i in population]
        Children = heavytail_children(population, mutation_rate, k, graph, n_edges, scores)
        population = Children
        xx3.append(generation)
        yy3.append(get_best_fitness(population, graph, n_edges))
        print(generation, get_best_fitness(population, graph, n_edges))
    return xx3, yy3

# 每个解是一个个体，
def main(args=get_args()):  # 优化目标是找到一组图的划分，使得cut_size最大
    print(args)
    yy1 = []
    xx1 = []
    graph, n_nodes, n_edges = graph_generator(args)
    np.random.seed(args.seed)  # 为下面调用的random.rand设置一个随机数种子
    # 返回一个1*n_nodes大小的由浮点数组成的随机数组
    # x在算法中扮演什么角色？ x在将图划分为两部分的过程中起辅助划分的作用
    # x是一个list，因此每个个体的表示形式也是一个list
    x = np.random.rand(n_nodes)  # 这里x使用实数值表示, 也可以直接使用01串表示, 并使用01串上的交叉变异算子，n_nodes是图中结点的数量
    # 在原始种群中只有1个个体，因此原始parent也只有一个（不需要刻意选择）
    best_fitness = get_fitness(graph, x, n_edges)
    for i in range(args.T):  # 简单的(1+1)ES
        print(i)
        tmp = x + np.random.randn(n_nodes) * args.sigma
        tmp_fitness = get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            yy1.append(best_fitness)
            xx1.append(i)
            print(i, best_fitness)

    xx2, yy2 = evolutionary_algorithm(n_nodes, 500, args.sigma, args.T, args.k, graph, n_edges)
    #xx2_1, yy2_1 = evolutionary_algorithm(n_nodes, 500, 0.1, args.T, args.k, graph, n_edges)
    xx3, yy3 = heavytail_evolutionary_algorithm(n_nodes, 500, args.sigma, args.T, args.k, graph, n_edges)
    #xx3_1, yy3_1 = heavytail_evolutionary_algorithm(n_nodes, 500, 0.1, args.T, args.k, graph, n_edges)
    # 画图
    plt.plot(xx1, yy1, label="(1+1)ES")
    plt.plot(xx2, yy2, label="(1+1)EA")
    plt.plot(xx3, yy3, label="(1+1)Fast GA")
    #plt.plot(xx2_1, yy2_1, label="(1+1)EA 0.1")
    #plt.plot(xx3_1, yy3_1, label="(1+1)Fast GA 0.1")
    plt.xlabel("times")
    plt.ylabel("fitness")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    runtime = end - start
    print('runtime', runtime)