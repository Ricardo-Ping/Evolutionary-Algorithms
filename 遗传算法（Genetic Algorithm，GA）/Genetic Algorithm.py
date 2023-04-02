import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

population_size = 25  # 种群数量
num_generations = 100  # 迭代次数
num_parents = 20  # 父代数量
num_bits = 20  # 每个个体包含的基因数量
crossover_rate = 0.9  # 交叉率
mutation_rate = 0.1  # 变异率
bounds = np.array([-1, 2])  # 搜索空间边界，即每个基因x的取值范围为[-5, 5]


# 初始化种群
def generate_initial_population(population_size, num_bits):
    population = []
    for i in range(population_size):
        temporary = []
        # 染色体暂存器
        for j in range(num_bits):
            temporary.append(random.choice([0, 1]))
            # 随机选择一个值，可以是0或1，将其添加到染色体中
        population.append(temporary)
        # 将染色体添加到种群中
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
def compute_fitness(decimal_population):
    fitness_values = []
    # 遍历每一个十进制数，计算适应度值并添加到适应度值列表中
    for decimal in decimal_population:
        # 调用适应度函数，计算适应度值
        y = fitness_function(decimal)
        # 将适应度值添加到适应度值列表中
        fitness_values.append(y)
    # 返回适应度值列表
    return fitness_values


# 选择操作
def selection(population, fitness_values, num_parents):
    # 保留适应度非负的个体,where返回的是一个二维数组
    positive_fitness_indices = np.where(np.array(fitness_values) >= 0)[0]
    # 根据下标找出个体和他的适应度值
    population = [population[i] for i in positive_fitness_indices]
    fitness_values = [fitness_values[i] for i in positive_fitness_indices]

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


# 变异操作
def mutation(offspring, mutation_rate):
    # 遍历每个后代
    for i in range(len(offspring)):
        # 遍历每个后代的基因
        for j in range(len(offspring[i])):
            # 判断是否进行变异操作
            if np.random.uniform(0, 1) <= mutation_rate:
                # 随机将基因进行变异
                offspring[i][j] = 1 - offspring[i][j]
    # 返回变异后的后代
    return offspring


if __name__ == '__main__':
    # 初始化种群
    population = generate_initial_population(population_size, num_bits)
    # 迭代num_generations轮
    best_fitness = float('-inf')
    best_individual = None
    best_fitnesses = []
    for generation in range(num_generations):
        # 二进制转换为十进制
        decimal_population = binary_to_decimal(population, bounds)
        # 计算适应度分数
        fitness_values = compute_fitness(decimal_population)
        # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
        next_generation = copy.deepcopy(population)
        # print('next_generation:', next_generation)
        # 选择父代个体
        parents = selection(population, fitness_values, num_parents)
        # 交叉操作
        offspring = single_point_crossover(parents, crossover_rate)
        # 变异操作
        offspring = mutation(offspring, mutation_rate)
        # 得到新的种群
        population = offspring
        # print('population:', population)

        # 找到当前一代中的最大适应度值的下标
        max_fitness_index = 0
        for i in range(1, len(fitness_values)):
            if fitness_values[i] > fitness_values[max_fitness_index]:
                max_fitness_index = i

        # 记录每一代的最好的适应度和个体
        # 适应度分数
        generation_best_fitness = fitness_values[max_fitness_index]
        # print('generation_best_fitness:',generation_best_fitness)
        # 适应度个体（十进制）
        generation_best_individual = decimal_population[max_fitness_index]
        # print('generation_best_individual:', generation_best_individual)
        best_fitnesses.append(generation_best_fitness)

        # 将每一代最好的适应度和个体放入原始种群
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

    # 将最佳个体的十进制值转换为二进制编码并输出
    best_individual_decimal = best_individual
    best_individual_binary = decimal_to_binary(best_individual_decimal, num_bits, bounds)

    print(
        f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal:.6f}, Best individual (binary): {best_individual_binary}")

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
