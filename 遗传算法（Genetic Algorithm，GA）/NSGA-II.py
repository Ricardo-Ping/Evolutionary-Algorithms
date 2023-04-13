"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/4/7 17:25
@File : NSGA-II.py
@function :多目标遗传算法
"""
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

population_size = 40  # 种群数量
num_generations = 1000  # 迭代次数
num_parents = 20  # 父代数量
bits_per_variable = 10  # 每个变量包含的基因数量
crossover_rate = 0.9  # 交叉率
mutation_rate = 0.1  # 变异率
bounds = np.array([[1, 10], [1, 10]])  # 搜索空间边界，即每个基因x的取值范围为[-1, 2]
# threshold = 1  # 阈值
num_variables = 2  # 问题的变量数


# 初始化种群，flag为1是二进制，flag为2是实数
def generate_initial_population(population_size, num_variables, bits_per_variable, flag):
    population = []
    for i in range(population_size):
        individual = []
        if flag == 1:
            for j in range(num_variables * bits_per_variable):
                individual.append(random.choice([0, 1]))
        else:
            for j in range(num_variables):
                individual.append(np.random.uniform(low=-1, high=2))
        population.append(individual)
    return population


# 实数编码适应度函数

# 目标函数1：f1(x, y) = 2x + y
def objective_function_1(individual):
    x, y = individual[:2]
    return 2 * x + y +3


# 目标函数2：f2(x, y) = x - y
def objective_function_2(individual):
    x, y = individual[:2]
    return x - y


# # 目标函数3：f3(x, y, z) = np.sin(x+y) + np.cos(y+z) + np.tan(z+x)
# def objective_function_3(individual):
#     x, y, z = individual
#     return np.sin(x + y) + np.cos(y + z) + np.tan(z + x)


# 二进制编码适应度函数

# 将二进制基因转换为实数
def binary_to_decimal(binary_individual, bounds, bits_per_variable):
    # num_variables = len(bounds)
    decimal_individual = []
    for i in range(num_variables):
        decimal_value = int(
            ''.join(str(bit) for bit in binary_individual[i * bits_per_variable:(i + 1) * bits_per_variable]), 2)
        decimal_individual.append(
            bounds[i][0] + (bounds[i][1] - bounds[i][0]) * decimal_value / (2 ** bits_per_variable - 1))
    return decimal_individual


# 二进制编码目标函数1：f1(x, y, z) = x^2 + y^2 + z^2
def objective_function_1_binary(binary_individual, bits_per_variable, bounds):
    decimal_individual = binary_to_decimal(binary_individual, bounds, bits_per_variable)
    return objective_function_1(decimal_individual)


# 二进制编码目标函数2：f2(x, y, z) = (x - 5)^2 + (y - 5)^2 + (z - 5)^2
def objective_function_2_binary(binary_individual, bits_per_variable, bounds):
    decimal_individual = binary_to_decimal(binary_individual, bounds, bits_per_variable)
    return objective_function_2(decimal_individual)


# # 二进制编码目标函数3：f3(x, y, z) = x * y * z
# def objective_function_3_binary(binary_individual, bits_per_variable, bounds):
#     decimal_individual = binary_to_decimal(binary_individual, bounds, bits_per_variable)
#     return objective_function_3(decimal_individual)


def dominates(solution1, solution2):
    '''
    dominates 函数：用于比较两个解决方案的适应度值，以确定一个解是否支配另一个解。
    如果解 a 在所有目标上都优于或等于解 b，并且在至少一个目标上优于解 b，则解 a 支配解 b
    :param solution1:
    :param solution2:
    :return: better_in_any_objective
    '''
    better_in_any_objective = False
    for value1, value2 in zip(solution1, solution2):
        if value1 > value2:
            return False
        elif value1 < value2:
            better_in_any_objective = True
    return better_in_any_objective


# def normalize_fitness_values(fitness_values, front):
#     front_values = np.array([fitness_values[i] for i in front])
#     min_values = np.min(front_values, axis=0)
#     max_values = np.max(front_values, axis=0)
#
#     # 检查分母是否为零
#     denominator = max_values - min_values
#     denominator[denominator == 0] = 1
#
#     normalized_values = (front_values - min_values) / denominator
#     return normalized_values


def normalize_fitness_values(fitness_values, front):
    front_values = np.array([fitness_values[i] for i in front])
    min_values = np.min(front_values, axis=0)
    max_values = np.max(front_values, axis=0)

    # 检查分母是否为零
    denominator = max_values - min_values
    denominator[denominator == 0] = 1

    normalized_values = (front_values - min_values) / denominator
    return normalized_values


def fast_non_dominated_sort(fitness_values):
    """
    快速非支配排序算法

    :param fitness_values: 一个包含多个目标适应度值的列表
    :return: 分层排序的结果
    """
    # 初始化
    dominated_solutions = [set() for _ in range(len(fitness_values))]
    dominating_solution_counts = [0] * len(fitness_values)
    fronts = [set()]

    # 遍历所有解
    for p, p_fitness in enumerate(fitness_values):
        for q, q_fitness in enumerate(fitness_values):
            if p == q:
                continue

            # 比较解 p 和解 q
            if dominates(p_fitness, q_fitness):
                dominated_solutions[p].add(q)
            elif dominates(q_fitness, p_fitness):
                dominating_solution_counts[p] += 1

        # 如果解 p 不被其他解支配，将其放入第一层
        if dominating_solution_counts[p] == 0:
            fronts[0].add(p)

    # 找到剩余的层
    i = 0
    while fronts[i]:
        next_front = set()
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                dominating_solution_counts[q] -= 1
                if dominating_solution_counts[q] == 0:
                    next_front.add(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(fitness_values, front):
    """
    计算拥挤距离

    :param fitness_values: 一个包含多个目标适应度值的列表
    :param front: 当前 Pareto 前沿
    :return: 拥挤距离的列表
    """
    normalized_fitness_values = normalize_fitness_values(fitness_values, front)
    front_values_dict = {i: normalized_fitness_values[j] for j, i in enumerate(front)}

    distances = [0] * len(front)
    num_objectives = len(fitness_values[0])

    for objective in range(num_objectives):
        sorted_front = sorted(front, key=lambda x: front_values_dict[x][objective])

        distances[0] = float('inf')
        distances[-1] = float('inf')

        for i in range(1, len(front) - 1):
            distances[i] += (front_values_dict[sorted_front[i + 1]][objective] - front_values_dict[sorted_front[i - 1]][
                objective])

    return distances



# 将 fronts 转换为 ranks
def convert_fronts_to_ranks(fronts):
    ranks = []
    for i, front in enumerate(fronts):
        for item in front:
            ranks.append((item, i))
    return ranks


# 使用二元锦标赛选择
def binary_tournament_selection(fronts, crowding_distances, num_parents):
    parents_indices = []

    for _ in range(num_parents):
        i1, i2 = random.sample(range(len(fronts)), 2)  # 修改这里，将 range(len(fronts)) 替换为 range(len(population))

        if fronts[i1] < fronts[i2]:
            parents_indices.append(i1)
        elif fronts[i1] > fronts[i2]:
            parents_indices.append(i2)
        else:
            if crowding_distances[i1] > crowding_distances[i2]:
                parents_indices.append(i1)
            else:
                parents_indices.append(i2)

    return parents_indices


# 汉明码距离
def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


# 欧式距离
def euclidean_distance(a, b):
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# 交叉函数
def crossover(parents, flag, crossover_rate, bounds):
    offspring = []
    num_parents = len(parents)

    if flag == 1:
        for i in range(0, num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            r = random.random()
            if r < crossover_rate:
                distance = hamming_distance(parent1, parent2)  # 使用汉明码距离
                threshold = random.randint(1, len(parent1))
                if distance < threshold:
                    crossover_point = random.randint(1, len(parent1) - 1)
                    child1 = parent1[:crossover_point] + parent2[crossover_point:]
                    child2 = parent2[:crossover_point] + parent1[crossover_point:]
                else:
                    child1 = parent1
                    child2 = parent2
            else:
                child1 = parent1
                child2 = parent2

            offspring.append(child1)
            offspring.append(child2)

    else:
        for i in range(0, num_parents, 2):
            parent1 = np.array(parents[i])
            parent2 = np.array(parents[i + 1])

            r = random.random()
            if r < crossover_rate:
                distance = euclidean_distance(parent1, parent2)  # 使用欧式距离
                threshold = np.random.rand(len(parent1)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                mask = distance < threshold
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2

            offspring.append(list(child1))
            offspring.append(list(child2))

    return offspring


# 变异函数
def mutation(offspring, flag, mutation_rate, bounds):
    mutated_offspring = []

    if flag == 1:
        for child in offspring:
            mutated_child = []
            for gene in child:
                r = random.random()
                if r < mutation_rate:
                    mutated_gene = 1 - gene  # 对于二进制编码，直接取反
                else:
                    mutated_gene = gene
                mutated_child.append(mutated_gene)
            mutated_offspring.append(mutated_child)
    else:
        for child in offspring:
            mutated_child = []
            for i, gene in enumerate(child):
                r = random.random()
                if r < mutation_rate:
                    mutated_gene = random.uniform(bounds[i, 0], bounds[i, 1])
                else:
                    mutated_gene = gene
                mutated_child.append(mutated_gene)
            mutated_offspring.append(mutated_child)

    return mutated_offspring


if __name__ == "__main__":
    # 初始化种群
    print('选择如何初始化种群，flag为1是二进制，flag为0是实数')
    flag = int(input())
    population = generate_initial_population(population_size, num_variables, bits_per_variable, flag)
    # 创建原始种群的深拷贝
    original_population = copy.deepcopy(population)

    # 根据flag值计算适应度值
    if flag == 1:  # 二进制编码
        fitness_values = [(
            objective_function_1_binary(individual, bits_per_variable, bounds),
            objective_function_2_binary(individual, bits_per_variable, bounds),
            # objective_function_3_binary(individual, bits_per_variable, bounds)
        ) for individual in population]
    else:  # 实数编码
        fitness_values = [(
            objective_function_1(individual),
            objective_function_2(individual),
            # objective_function_3(individual)
        ) for individual in population]
    # print("fitness_values:", fitness_values)

    # 存储每一代的最佳个体和适应度，Pareto 前沿存储信息
    best_individuals = []
    best_fitnesses = []
    pareto_fronts = []

    # 进化过程
    for generation in range(num_generations):
        # 非支配排序
        fronts = fast_non_dominated_sort(fitness_values)
        # print("Fronts:", fronts)

        # 计算拥挤距离
        crowding_distances = []
        for front in fronts:
            crowding_distances.extend(crowding_distance(fitness_values, front))
        print(fronts)

        # 将 fronts 转换为 ranks
        ranks = convert_fronts_to_ranks(fronts)

        # 选择父代个体
        parent_indices = binary_tournament_selection(ranks, crowding_distances, num_parents)
        parents = [population[i] for i in parent_indices]

        # 交叉操作
        offspring = crossover(parents, flag, crossover_rate, bounds)

        # 变异操作
        offspring = mutation(offspring, flag, mutation_rate, bounds)

        # 合并父代和子代
        combined_population = population + offspring
        if flag == 1:  # 二进制编码
            combined_fitness_values = [(
                objective_function_1_binary(individual, bits_per_variable, bounds),
                objective_function_2_binary(individual, bits_per_variable, bounds),
                # objective_function_3_binary(individual, bits_per_variable, bounds)
            ) for individual in combined_population]
        else:  # 实数编码
            combined_fitness_values = [(
                objective_function_1(individual),
                objective_function_2(individual),
                # objective_function_3(individual)
            ) for individual in combined_population]

        # 非支配排序和拥挤距离计算
        combined_fronts = fast_non_dominated_sort(combined_fitness_values)
        print("combined_fronts:",combined_fronts)
        combined_crowding_distances = []
        for front in combined_fronts:
            combined_crowding_distances.extend(crowding_distance(combined_fitness_values, front))

        # 选择新的种群
        new_population_indices = []

        # 确定精英个体数目
        elite_percentage = 0.1
        num_elites = int(population_size * elite_percentage)
        print("num_elites:",num_elites)

        # 找到上一代种群中的精英个体
        elite_indices = sorted(range(len(crowding_distances)), key=lambda i: -crowding_distances[i])[:num_elites]
        elites = [original_population[i] for i in elite_indices]
        print("elites:",elites)

        # 添加精英个体到新种群
        new_population_indices.extend(elite_indices)

        for front in combined_fronts:
            if len(new_population_indices) + len(front) <= population_size:
                # 如果当前层的个体数目加上已选择的个体数目不超过种群大小，则将该层全部选择
                for i in front:
                    if i not in elite_indices:
                        new_population_indices.append(i)
            else:
                # 如果加上当前层个体数目会超过种群大小，则使用拥挤距离选择剩余个体
                remaining_indices = sorted(front, key=lambda i: -combined_crowding_distances[i])
                remaining_indices = [i for i in remaining_indices if i not in elite_indices]
                new_population_indices.extend(remaining_indices[:population_size - len(new_population_indices)])
                break

    #     # 记录每一代的最佳个体和适应度
    #     best_individual_index = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
    #     best_individual = original_population[best_individual_index]
    #     best_fitness = fitness_values[best_individual_index]
    #     best_individuals.append(best_individual)
    #     best_fitnesses.append(best_fitness)
    #
    #     # 输出每一代的最佳个体和适应度
    #     print("第{}代最佳个体:".format(generation + 1), best_individuals[-1])
    #     print("第{}代最佳适应度:".format(generation + 1), best_fitnesses[-1])
    #     print("")
    #
        # 更新种群和适应度值
        population = [combined_population[i] for i in new_population_indices]
        fitness_values = [combined_fitness_values[i] for i in new_population_indices]

        # 更新原始种群
        original_population = copy.deepcopy(population)

        # 添加 Pareto 前沿到 pareto_fronts 列表
        pareto_fronts.append([original_population[i] for i in fronts[0]])
        # print("pareto_fronts:", pareto_fronts)

    #
    # # 输出最终最佳个体和适应度
    # final_best_individual = best_individuals[-1]
    # final_best_fitness = best_fitnesses[-1]
    #
    # print("最终最佳个体:", final_best_individual)
    # print("最终最佳适应度:", final_best_fitness)
    #
    # # 画出最佳适应度曲线
    # plt.plot(best_fitnesses)  # 修改此行以显示最佳适应度值
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness")
    # plt.title("Evolution of Best Fitness")
    # plt.show()
    # Pareto 前沿的最后一代
    final_pareto_front = pareto_fronts[-1]

    # 计算目标值
    objective_values_f1 = [objective_function_1(individual) for individual in final_pareto_front]
    objective_values_f2 = [objective_function_2(individual) for individual in final_pareto_front]

    # 绘制 Pareto 前沿散点图
    plt.scatter(objective_values_f1, objective_values_f2)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Front")
    plt.show()



