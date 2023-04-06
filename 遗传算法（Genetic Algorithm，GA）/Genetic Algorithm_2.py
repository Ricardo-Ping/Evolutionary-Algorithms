# -*- coding: utf-8 -*-
# @Time : 2023/4/6 15:01
# @Author : Ricardo_PING
# @File : Genetic Algorithm_2
# @Project : Genetic Algorithm.py
# @Function ：在之前的遗传算法基础上加上三个功能：1.选择实数或者二进制编码 2.进行两点交叉或者算数交叉 3.进行近亲交叉回避（计算汉明码距离和欧式距离）
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import heapq

population_size = 30  # 种群数量
num_generations = 10000  # 迭代次数
num_parents = 20  # 父代数量
num_bits = 20  # 每个个体包含的基因数量
crossover_rate = 0.9  # 交叉率
mutation_rate = 0.1  # 变异率
bounds = np.array([-1, 2])  # 搜索空间边界，即每个基因x的取值范围为[-1, 2]
threshold = 5  # 阈值


# 初始化种群，flag为1是二进制，flag为2是实数
def generate_initial_population(population_size, num_bits, flag):
    population = []
    for i in range(population_size):
        temporary = []
        for j in range(num_bits):
            if flag == 1:
                temporary.append(random.choice([0, 1]))
            else:
                temporary.append(np.random.uniform(low=-1, high=2))
        population.append(temporary)
    return population


# 从二进制编码转换为十进制数值
def binary_to_decimal(population, bounds):
    decimal_population = []  # 存储所有染色体的十进制数值
    for chromosome in population:  # 遍历种群中的每个染色体
        decimal = 0  # 初始化十进制数值
        for i, gene in enumerate(chromosome):  # 遍历染色体中的每个基因
            decimal += gene * (2 ** i)  # 将基因的值乘以2的幂次方，求和得到十进制数值
        lower_bound, upper_bound = bounds[0], bounds[1]
        mapped_decimal = lower_bound + (decimal / ((2 ** len(chromosome)) - 1)) * (upper_bound - lower_bound)
        decimal_population.append(mapped_decimal)  # 将映射后的十进制数值添加到列表中
    return decimal_population  # 返回所有染色体映射后的十进制数值列表


'''
1.用 (decimal_value - min_bound) 计算该值在范围内的位置
2.将该位置除以 (max_bound - min_bound)，得到一个比例，表示该位置在范围内所占比例
3.乘以 (2 ** num_bits - 1) 得到一个整数范围内的位置
4.用 int() 函数将该位置转换为整数
'''


# 十进制数值转换为二进制编码
def decimal_to_binary(decimal_value, num_bits, bounds):
    # 将十进制数值映射到整数范围
    min_bound, max_bound = bounds
    fixed_point_value = int((decimal_value - min_bound) / (max_bound - min_bound) * (2 ** num_bits - 1))

    # 将整数转换为二进制编码列表
    # format() 函数将整数转换为二进制字符串，并指定了字符串的位数
    binary_value = [int(bit) for bit in format(fixed_point_value, f'0{num_bits}b')]

    return binary_value


# 适应度函数
def fitness_function(x):
    # 计算目标函数值
    y = x * np.sin(10 * math.pi * x) + 2.0
    # 返回适应度（即目标函数值）
    return y


# 适应度分数
def compute_fitness(population, flag):
    fitness_values = []

    if flag == 1:  # 如果是二进制的话传入的是经过转换为10进制的数字数组
        # 遍历每一个十进制数，计算适应度值并添加到适应度值列表中
        for decimal in population:
            # 调用适应度函数，计算适应度值
            y = fitness_function(decimal)
            # 将适应度值添加到适应度值列表中
            fitness_values.append(y)
    else:  # 如果是实数的话传入的应该是一个二维数组
        # 遍历二维数组中的每一个个体
        for individual in population:
            individual_fitness_values_sum = 0  # 实数中单个个体的适应度总和
            individual_fitness_values = []
            # 遍历单个个体中的每一个十进制数
            for decimal in individual:
                # 调用适应度函数，计算适应度值
                y = fitness_function(decimal)
                individual_fitness_values.append(y)
                individual_fitness_values_sum += y

            # 找到 individual_fitness_values 中最大的 num_bits/10个元素
            top_values = heapq.nlargest(int(num_bits / 20), individual_fitness_values)

            # 计算最大的 num_bits/10个元素的平均值
            average_top = sum(top_values) / len(top_values)

            # 将适应度值添加到适应度值列表中
            fitness_values.append(average_top)

    # 返回适应度值列表
    return fitness_values


# 选择操作
def selection(population, fitness_values, num_parents, flag):
    if flag == 1:
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
    else:
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
        num_parents = 2 * (population_size // 2)  # 确保num_parents是偶数
        for i in range(num_parents):
            # 产生一个0到1之间的随机数
            rand_num = np.random.uniform(low=0, high=1.0)
            # 确定随机数出现在哪个个体的概率区域内
            for j in range(len(cumulative_probabilities)):
                # 当前随机数小于等于累积概率列表中的某个元素，就选择该元素对应的个体作为父代
                if rand_num <= cumulative_probabilities[j]:
                    parents.append(population[j])  # 返回整个个体
                    break

    # print("parents:",parents)
    return parents


# 单点交叉操作
def single_point_crossover(parents, crossover_rate):
    offspring = []  # 初始化后代列表
    num_parents = len(parents)  # 父代数量
    num_bits = len(parents[0])  # 每个父代的二进制编码位数

    # 对每两个相邻的父代进行交叉操作
    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]  # 第一个父代
        parent2 = parents[i + 1]  # 第二个父代

        # 随机生成交叉点
        crossover_point = np.random.randint(1, num_bits)

        # 根据交叉率进行交叉操作
        if np.random.random() < crossover_rate:
            # 生成新的后代
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            # 如果不交叉，则直接将父代作为后代
            offspring1 = parent1
            offspring2 = parent2

        # 将后代添加到列表中
        offspring.append(offspring1)
        offspring.append(offspring2)

    return offspring  # 返回生成的后代列表


# 计算汉明码距离
def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


# 计算欧式距离
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# 带距离的交叉
def crossover_with_avoidance(parents, crossover_rate, threshold, flag):
    offspring = []
    num_parents = len(parents)

    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        if flag == 1:
            distance = hamming_distance(parent1, parent2)
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
                alpha = np.random.uniform(0, 1, size=len(parent1))
                offspring1 = [alpha[i] * parent1[i] + (1 - alpha[i]) * parent2[i] for i in range(len(parent1))]
                offspring2 = [alpha[i] * parent2[i] + (1 - alpha[i]) * parent1[i] for i in range(len(parent1))]
            else:
                offspring1 = parent1
                offspring2 = parent2

        offspring.append(offspring1)
        offspring.append(offspring2)
        '''
                使用算术交叉（Arithmetic crossover）作为另一种交叉方式。算术交叉方法根据给定的权重值组合父代基因，生成子代基因。
                算术交叉的一个优点是它可以在基因表示为实数时很好地保留父代的基因特征，同时在子代中产生新的特征组合。
                生成一个与父代基因长度相同的随机权重值列表alpha。然后，我们通过遍历父代基因并计算加权平均值来生成两个子代基因。
                这种方法适用于实数编码的遗传算法，因为它提供了一种平滑地在父代基因之间插值的方法，可以在搜索空间内生成新的解。
                使用算术交叉还可以在某种程度上保持父代基因的特征，同时在子代中引入新的特征组合。
                '''

    return offspring


# 双点交叉操作
def two_point_crossover(parents, crossover_rate):
    offspring = []  # 初始化后代列表
    num_parents = len(parents)  # 父代数量
    num_bits = len(parents[0])  # 每个父代的位数

    # 对每两个相邻的父代进行交叉操作
    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]  # 第一个父代
        parent2 = parents[i + 1]  # 第二个父代

        # 随机生成两个交叉点
        crossover_point1 = np.random.randint(1, num_bits)
        crossover_point2 = np.random.randint(crossover_point1, num_bits)

        # 根据交叉率进行交叉操作
        if np.random.random() < crossover_rate:
            # 生成新的后代
            offspring1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[
                                                                                                   crossover_point2:]
            offspring2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[
                                                                                                   crossover_point2:]
        else:
            # 如果不交叉，则直接将父代作为后代
            offspring1 = parent1
            offspring2 = parent2

        # 将后代添加到列表中
        offspring.append(offspring1)
        offspring.append(offspring2)

    return offspring  # 返回生成的后代列表


# 变异操作
def mutation(offspring, mutation_rate, flag):
    # 遍历每个后代
    for i in range(len(offspring)):
        # 遍历每个后代的基因
        for j in range(len(offspring[i])):
            # 判断是否进行变异操作
            if np.random.uniform(0, 1) <= mutation_rate:
                if flag == 1:
                    # 二进制编码：随机将基因进行变异
                    offspring[i][j] = 1 - offspring[i][j]
                # else:
                #     # 实数编码：基因上下波动
                #     # direction = random.choice([-1, 1])
                #     direction = random.uniform(-1, 2)
                #     offspring[i][j] += direction*0.1
                #     # 如果需要限制实数范围，可以在这里添加条件(根据情况进行修改)
                #     offspring[i][j] = min(max(offspring[i][j], -1), 2)
                else:
                    # 实数编码：随机选择一位或多位
                    num_to_replace = np.random.randint(1, len(offspring[i]) + 1)
                    positions_to_replace = np.random.choice(len(offspring[i]), num_to_replace, replace=False)

                    for pos in positions_to_replace:
                        # 用随机产生的新值代替原有的实数值
                        offspring[i][pos] = random.uniform(-1, 2)
                        # 如果需要限制实数范围，可以在这里添加条件(根据情况进行修改)
                        # offspring[i][pos] = min(max(offspring[i][pos], -1), 2)
    # 返回变异后的后代
    return offspring


if __name__ == '__main__':
    print('选择如何初始化种群，flag为1是二进制，flag为0是实数')
    flag = int(input())
    # 初始化种群 生产的是population_size*num_bits的二维数组
    population = generate_initial_population(population_size, num_bits, flag)

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

            # 选择父代个体
            parents = selection(population, fitness_values, num_parents, flag)

            # 交叉操作
            offspring = crossover_with_avoidance(parents, crossover_rate, threshold, flag)
            # print("offspring:", offspring)
            # 变异操作
            after_offspring = mutation(offspring, mutation_rate, flag)
            # print("after_offspring:", after_offspring)

            # 得到新的种群
            population = after_offspring

            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = 0
            for i in range(1, len(fitness_values)):
                if fitness_values[i] > fitness_values[max_fitness_index]:
                    max_fitness_index = i

            # 记录每一代的最好地适应度和个体

            # 适应度分数
            generation_best_fitness = fitness_values[max_fitness_index]

            # 适应度个体（十进制）
            generation_best_individual = decimal_population[max_fitness_index]

            best_fitnesses.append(generation_best_fitness)

            # 将每一代最好地适应度和个体放入原始种群
            population[0] = next_generation[max_fitness_index]

            # 输出最佳个体的二进制编码和映射后的十进制值
            best_individual_binary = decimal_to_binary(generation_best_individual, num_bits, bounds)
            print(
                f"Generation {generation + 1} - Best fitness: {generation_best_fitness:.6f}, Best individual - Binary: {best_individual_binary}, Decimal: {generation_best_individual:.6f}")

            # 更新全局最优解
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best_individual

                # 如果找到了Best fitness大于 3.85027，就退出循环
                if generation_best_fitness > 3.85027:
                    print(f"Solution found after {generation + 1} generations.")
                    break
        else:
            # 计算适应度分数
            fitness_values = compute_fitness(population, flag)
            # print(fitness_values)
            # print(len(fitness_values))

            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            # print("next_generation:",next_generation)

            # 选择父代个体
            parents = selection(population, fitness_values, num_parents, flag)

            # 交叉操作
            offspring = crossover_with_avoidance(parents, crossover_rate, threshold, flag)
            # print("offspring:",offspring)
            # 变异操作
            after_offspring = mutation(offspring, mutation_rate, flag)
            # print("after_offspring:", after_offspring)

            # 得到新的种群
            population = after_offspring

            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = 0
            for i in range(1, len(fitness_values)):
                # 假设 fitness_values 是二维数组，我们需要比较适应度值之和
                if np.sum(fitness_values[i]) > np.sum(fitness_values[max_fitness_index]):
                    max_fitness_index = i
            # print("max_fitness_index:",max_fitness_index)

            # 记录每一代的最好地适应度和个体

            # 适应度分数
            generation_best_fitness = np.sum(fitness_values[max_fitness_index])
            # print("generation_best_fitness:",generation_best_fitness)

            # 适应度个体
            generation_best_individual = population[max_fitness_index]
            # print("generation_best_individual:",generation_best_individual)
            # print(len(generation_best_individual))

            best_fitnesses.append(generation_best_fitness)
            # print("best_fitnesses:",best_fitnesses)

            # 将每一代最好地适应度和个体放入原始种群
            population[0] = next_generation[max_fitness_index]
            # print("population[0]:",population[0])

            print(
                f"Generation {generation + 1} - Best fitness: {generation_best_fitness:.6f}, Best individual: {generation_best_individual}")

            # 更新全局最优解
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best_individual

                # 如果找到了Best fitness大于 3.85027，就退出循环
                if generation_best_fitness > 3.85027:
                    print(f"Solution found after {generation + 1} generations.")
                    break

    if flag == 1:
        # 将最佳个体的十进制值转换为二进制编码并输出
        best_individual_decimal = best_individual
        best_individual_binary = decimal_to_binary(best_individual_decimal, num_bits, bounds)
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal:.6f}, Best individual (binary): {best_individual_binary}")
    else:
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual: {', '.join([f'{x:.6f}' for x in best_individual])}")

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
