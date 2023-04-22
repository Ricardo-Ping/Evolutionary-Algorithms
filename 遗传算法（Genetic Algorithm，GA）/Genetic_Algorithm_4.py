# -*- coding: utf-8 -*-
# @Time : 2023/4/13 22:05
# @Author : Ricardo_PING
# @File : 5
# @Project : 遗传算法（Genetic Algorithm，GA）
# @Function ：

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import heapq
from itertools import chain

population_size = 20  # 种群数量
num_generations = 10000  # 迭代次数
num_parents = 20  # 父代数量
num_bits = 10  # 每个个体包含的基因数量
crossover_rate = 0.9  # 交叉率
mutation_rate = 0.1  # 变异率
threshold = 1  # 阈值

# Problem parameters
num_variables = 2
bounds = [(-10, 10), (-10, 10)]


# 初始化种群，flag为1是二进制，flag为2是实数
def generate_initial_population(population_size, num_variables, flag):
    population = []
    for i in range(population_size):
        individual = []
        for j in range(num_variables):
            if flag == 1:  # binary encoding
                individual.append([random.choice([0, 1]) for _ in range(num_bits)])
            else:  # real encoding
                individual.append(np.random.uniform(low=bounds[j][0], high=bounds[j][1]))
        population.append(individual)
    return population


# 从二进制编码转换为十进制数值
def binary_to_decimal(population, bounds):
    decimal_population = []  # 存储所有染色体的十进制数值
    for individual in population:  # 遍历种群中的每个个体
        decimals = [[], []]  # 初始化包含两个变量的十进制数值列表
        for i, variable in enumerate(individual):  # 遍历个体中的每个变量
            decimal = 0  # 初始化十进制数值
            for j, gene in enumerate(variable):  # 遍历变量中的每个基因
                decimal += gene * (2 ** j)  # 将基因的值乘以2的幂次方，求和得到十进制数值
            lower_bound, upper_bound = bounds[i]
            mapped_decimal = lower_bound + (decimal / ((2 ** len(variable)) - 1)) * (upper_bound - lower_bound)
            decimals[i].append(mapped_decimal)  # 将映射后的十进制数值添加到列表中
        decimal_population.append(decimals)  # 将包含两个变量的十进制数值列表添加到总列表中
    # print("decimal_population:",decimal_population)
    return decimal_population  # 返回所有染色体映射后的十进制数值列表


# 十进制数值转换为二进制编码
def decimal_to_binary(decimal_values, num_bits, bounds):
    binary_values = []
    for i, decimal_value_list in enumerate(decimal_values):
        min_bound, max_bound = bounds[i]
        decimal_value = decimal_value_list[0]

        # 将十进制数值映射到整数范围
        fixed_point_value = int((decimal_value - min_bound) / (max_bound - min_bound) * (2 ** num_bits - 1))

        # 将整数转换为二进制编码列表
        # format() 函数将整数转换为二进制字符串，并指定了字符串的位数
        binary_value = [int(bit) for bit in format(fixed_point_value, f'0{num_bits}b')]
        binary_values.append(binary_value)

    return binary_values


# 适应度函数
def fitness_function(individual):
    if flag == 1:
        x1, x2 = individual[0][0], individual[1][0]  # 解析每个变量的值
    else:
        x1, x2 = individual[0], individual[1]
    y = 3 * (x1 ** 2 - x2) ** 2
    return y


# 适应度分数
def compute_fitness(population, flag):
    # print(population)
    fitness_values = []
    if flag == 1:
        # 遍历每一个十进制数，计算适应度值并添加到适应度值列表中
        for individual in population:
            # 调用适应度函数，计算适应度值
            y = fitness_function(individual)
            # 将适应度值添加到适应度值列表中
            fitness_values.append(y)
    else:
        # 遍历每一个十进制数，计算适应度值并添加到适应度值列表中
        for individual in population:
            # 调用适应度函数，计算适应度值
            y = fitness_function(individual)
            # 将适应度值添加到适应度值列表中
            fitness_values.append(y)
    # 返回适应度值列表
    # print("fitness_values:",fitness_values)
    return fitness_values


# 选择操作
def selection(population, fitness_values, num_parents, flag):
    # 保留适应度非负的个体,where返回的是一个二维数组
    positive_fitness_indices = np.where(np.array(fitness_values) >= 0)[0]
    # 根据下标找出个体和他的适应度值
    population = [population[i] for i in positive_fitness_indices]
    # print('population:',population)
    fitness_values = [fitness_values[i] for i in positive_fitness_indices]
    # print('fitness_values:',fitness_values)
    # 计算适应度总和
    fitness_sum = sum(fitness_values)
    # 计算每个个体的选择概率，与适应度分数成正比
    probabilities = [fitness_value / fitness_sum for fitness_value in fitness_values]
    # 计算累积概率分布
    cumulative_probabilities = np.cumsum(probabilities)
    # 选择父代个体
    parents = []
    for i in range(num_parents):
        # 产生一个0到1之间的随机数
        rand_num = np.random.uniform(low=0, high=1.0)
        # 确定随机数出现在哪个个体的概率区域内
        for j in range(len(cumulative_probabilities)):
            # 当前随机数小于等于累积概率列表中的某个元素，就选择该元素对应的个体作为父代
            if rand_num <= cumulative_probabilities[j]:
                parents.append(population[j])  # 直接返回基因
                break
    # print("parents:",parents)
    return parents


# 计算汉明码距离
def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


# 计算欧式距离

def euclidean_distance(u, v):
    u = np.array(u)
    v = np.array(v)
    dist = np.linalg.norm(u - v)  # 求向量差的范数，即欧式距离
    return dist


# 带距离的交叉
def crossover_with_avoidance(parents, crossover_rate, threshold, flag):
    offspring = []
    num_parents = len(parents)
    # print(parents)

    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        if flag == 1:
            distance = hamming_distance(parent1, parent2)
            # print(distance)
            if np.random.random() < crossover_rate and distance > threshold:
                crossover_point = np.random.randint(1, len(parent1))
                offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
                offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                offspring1 = parent1
                offspring2 = parent2
        else:
            distance = euclidean_distance(parent1, parent2)
            threshold = 1
            if np.random.random() < crossover_rate and distance > threshold:
                alpha1 = np.random.uniform(0, 1)
                alpha2 = np.random.uniform(0, 1)
                offspring1 = [alpha1 * parent1[0] + (1 - alpha1) * parent2[0],
                              alpha2 * parent1[1] + (1 - alpha2) * parent2[1]]
                offspring2 = [alpha1 * parent2[0] + (1 - alpha1) * parent1[0],
                              alpha2 * parent2[1] + (1 - alpha2) * parent1[1]]

            else:
                offspring1 = parent1
                offspring2 = parent2

        # print(offspring1)
        # print(offspring2)
        offspring.append(offspring1)
        offspring.append(offspring2)

    # print("offspring:",offspring)
    return offspring


# 变异操作
def mutation(offspring, mutation_rate, flag, down, up):
    # 遍历每个后代
    for i in range(len(offspring)):
        # 遍历每个后代的基因
        for j in range(len(offspring[i])):
            # 判断是否进行变异操作
            if np.random.uniform(0, 1) <= mutation_rate:
                if flag == 1:
                    # 二进制编码：随机将基因进行变异
                    offspring[i][j] = 1 - offspring[i][j]
                else:
                    # 实数编码：随机选择一位或多位
                    num_to_replace = np.random.randint(1, len(offspring[i]) + 1)
                    positions_to_replace = np.random.choice(len(offspring[i]), num_to_replace, replace=False)

                    for pos in positions_to_replace:
                        # 用随机产生的新值代替原有的实数值
                        new_value = offspring[i][pos] * np.random.choice([0.7, 1.3])
                        # 判断是否超出范围
                        if new_value < down:
                            offspring[i][pos] = down
                        elif new_value > up:
                            offspring[i][pos] = up
                        else:
                            offspring[i][pos] = new_value
    # 返回变异后的后代
    return offspring


if __name__ == '__main__':
    print('选择如何初始化种群，flag为1是二进制，flag为0是实数')
    flag = int(input())
    # 初始化种群 生产的是population_size*num_bits的二维数组
    population = generate_initial_population(population_size, num_variables, flag)

    # 迭代num_generations轮
    best_fitness = float('-inf')
    best_individual = None
    best_fitnesses = []

    for generation in range(num_generations):
        if flag == 1:
            # 二进制转换为十进制
            decimal_population = binary_to_decimal(population, bounds)
            # 计算适应度分数
            fitness_values = compute_fitness(decimal_population, flag)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            # print("population:",population)
            merged_population = [individual[0] + individual[1] for individual in population]
            # print("merged_population:", merged_population)

            # 选择父代个体
            parents = selection(merged_population, fitness_values, num_parents, flag)
            # print("parents:", parents)

            # 交叉操作
            offspring = crossover_with_avoidance(parents, crossover_rate, threshold, flag)
            # print("offspring:", offspring)

            # 变异操作
            after_offspring = mutation(offspring, mutation_rate, flag, -10, 10)
            # print("after_offspring:", after_offspring)

            split_offspring = []
            for individual in after_offspring:
                x1 = individual[:len(individual) // 2]
                x2 = individual[len(individual) // 2:]
                split_offspring.append([x1, x2])

            # print("split_offspring:", split_offspring)

            # 得到新的种群
            population = split_offspring

            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = 0
            for i in range(1, len(fitness_values)):
                if fitness_values[i] > fitness_values[max_fitness_index]:
                    max_fitness_index = i

            # 记录每一代的最好地适应度和个体

            # 适应度分数
            generation_best_fitness = fitness_values[max_fitness_index]
            # print(generation_best_fitness)

            # 适应度个体（十进制）
            generation_best_individual = decimal_population[max_fitness_index]
            # print(generation_best_individual)

            best_fitnesses.append(generation_best_fitness)

            # 将每一代最好地适应度和个体放入原始种群
            population[0] = next_generation[max_fitness_index]
            # print(population[0])

            # 输出最佳个体的二进制编码和映射后的十进制值
            best_individual_binary = decimal_to_binary(generation_best_individual, num_bits, bounds)
            # print(best_individual_binary)
            flattened_individual = list(chain.from_iterable(generation_best_individual))
            print(
                f"Generation {generation + 1} - Best fitness: {generation_best_fitness:.6f}, Best individual - Binary: {best_individual_binary}, Decimal: {', '.join([f'{x:.6f}' for x in flattened_individual])}")
            # 更新全局最优解
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best_individual

                # 如果找到了Best fitness大于 36300，就退出循环
                if generation_best_fitness >= 36300:
                    print(f"Solution found after {generation + 1} generations.")
                    break
        else:
            # 计算适应度分数
            fitness_values = compute_fitness(population, flag)

            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)

            # 选择父代个体
            parents = selection(population, fitness_values, num_parents, flag)

            # 交叉操作
            offspring = crossover_with_avoidance(parents, crossover_rate, threshold, flag)

            # 变异操作
            after_offspring = mutation(offspring, mutation_rate, flag, -10, 10)

            # 得到新的种群
            population = after_offspring

            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = fitness_values.index(max(fitness_values))

            # 计算新种群（经过选择、交叉和变异操作后的种群）的适应度分数
            new_fitness_values = compute_fitness(population, flag)

            # 找到新种群中的最小适应度值的下标
            min_fitness_index = new_fitness_values.index(min(new_fitness_values))

            # 记录每一代的最好地适应度和个体
            generation_best_fitness = fitness_values[max_fitness_index]
            generation_best_individual = next_generation[max_fitness_index]

            best_fitnesses.append(generation_best_fitness)

            # 用原始种群（next_generation）中具有最大适应度值的个体替换新种群（population）中具有最小适应度值的个体
            population[min_fitness_index] = next_generation[max_fitness_index]

            print(
                f"Generation {generation + 1} - Best fitness: {generation_best_fitness:.6f}, Best individual: {generation_best_individual}")

            # 更新全局最优解
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best_individual

                # 如果找到了Best fitness大于 3.85027，就退出循环
                if generation_best_fitness >= 36300:
                    print(f"Solution found after {generation + 1} generations.")
                    break

    if flag == 1:
        # 将最佳个体的十进制值转换为二进制编码并输出
        best_individual_decimal = best_individual
        best_individual_binary = decimal_to_binary(best_individual_decimal, num_bits, bounds)
        flattened_best_individual_binary = list(chain.from_iterable(best_individual_binary))
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal}, Best individual (binary): {', '.join([str(x) for x in flattened_best_individual_binary])}")
    else:
        flat_individual = [str(x) for x in best_individual]

        # 输出最终结果
        print(f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual: {', '.join(flat_individual)}")

    # 绘制每次迭代的最佳适应度
    plt.plot(best_fitnesses, label='Best fitness per generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')

    # 标记全局最优解
    best_generation = best_fitnesses.index(best_fitness)
    plt.plot(best_generation, best_fitness, 'ro', label='Global best solution')

    # 显示图例和图形
    plt.legend()
    plt.show()
