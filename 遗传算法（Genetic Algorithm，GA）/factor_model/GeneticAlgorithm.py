# -*- coding: utf-8 -*-
# @Time : 2023/4/20 15:35
# @Author : Ricardo_PING
# @File : GeneticAlgorithm
# @Project : Genetic Algorithm.py
# @Function ：

# GeneticAlgorithm.py
import copy
import random
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
from fitness_function_factory import FitnessFunctionFactory


class GeneticAlgorithm:
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate,
                 threshold, bounds, fitness_function_type):
        # Initialize class variables here
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.threshold = threshold
        self.bounds = bounds
        self.num_bits = num_bits
        self.fitness_function_instance = FitnessFunctionFactory.create_fitness_function(fitness_function_type)

    def generate_initial_population(self):
        pass

    # 适应度函数
    def fitness_function(self, individual):
        x1, x2 = individual[0][0], individual[1][0]
        return self.fitness_function_instance.calculate(x1, x2)

    # 适应度分数
    def compute_fitness(self, population):
        fitness_values = []
        for individual in population:
            y = self.fitness_function(individual)
            fitness_values.append(y)
        return fitness_values

    # 选择操作
    def selection(self, population, fitness_values):
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
        for i in range(self.num_parents):
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

    # 带距离的交叉
    def crossover_with_avoidance(self, parents):
        pass

    def mutation(self, offspring, down, up):
        pass

    def run(self):
        pass


class GeneticAlgorithmBinary(GeneticAlgorithm):
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate, threshold, bounds, fitness_function_type):
        super().__init__(population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                         mutation_rate, threshold, bounds, fitness_function_type)

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            individual = []
            for j in range(self.num_variables):
                individual.append([random.choice([0, 1]) for _ in range(self.num_bits)])
            population.append(individual)
        return population

    # 从二进制编码转换为十进制数值
    def binary_to_decimal(self, population):
        decimal_population = []  # 存储所有染色体的十进制数值
        for individual in population:  # 遍历种群中的每个个体
            decimals = [[], []]  # 初始化包含两个变量的十进制数值列表
            for i, variable in enumerate(individual):  # 遍历个体中的每个变量
                decimal = 0  # 初始化十进制数值
                for j, gene in enumerate(variable):  # 遍历变量中的每个基因
                    decimal += gene * (2 ** j)  # 将基因的值乘以2的幂次方，求和得到十进制数值
                lower_bound, upper_bound = self.bounds[i]
                mapped_decimal = lower_bound + (decimal / ((2 ** len(variable)) - 1)) * (upper_bound - lower_bound)
                decimals[i].append(mapped_decimal)  # 将映射后的十进制数值添加到列表中
            decimal_population.append(decimals)  # 将包含两个变量的十进制数值列表添加到总列表中
        # print("decimal_population:",decimal_population)
        return decimal_population  # 返回所有染色体映射后的十进制数值列表

    # 十进制数值转换为二进制编码
    def decimal_to_binary(self, decimal_values):
        binary_values = []
        for i, decimal_value_list in enumerate(decimal_values):
            min_bound, max_bound = self.bounds[i]
            decimal_value = decimal_value_list[0]

            # 将十进制数值映射到整数范围
            fixed_point_value = int((decimal_value - min_bound) / (max_bound - min_bound) * (2 ** self.num_bits - 1))

            # 将整数转换为二进制编码列表
            # format() 函数将整数转换为二进制字符串，并指定了字符串的位数
            binary_value = [int(bit) for bit in format(fixed_point_value, f'0{self.num_bits}b')]
            binary_values.append(binary_value)

        return binary_values

    # 计算汉明码距离
    def hamming_distance(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    # 带距离的交叉
    def crossover_with_avoidance(self, parents):
        offspring = []
        num_parents = len(parents)

        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            distance = self.hamming_distance(parent1, parent2)
            # print(distance)
            if np.random.random() < self.crossover_rate and distance > self.threshold:
                crossover_point = np.random.randint(1, len(parent1))
                offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
                offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                offspring1 = parent1
                offspring2 = parent2

            offspring.append(offspring1)
            offspring.append(offspring2)

        return offspring

    def mutation(self, offspring, down, up):
        # 遍历每个后代
        for i in range(len(offspring)):
            # 遍历每个后代的基因
            for j in range(len(offspring[i])):
                # 判断是否进行变异操作
                if np.random.uniform(0, 1) <= self.mutation_rate:
                    # 二进制编码：随机将基因进行变异
                    offspring[i][j] = 1 - offspring[i][j]
        # 返回变异后的后代
        return offspring

    def run(self):
        # 初始化种群 生产的是population_size*num_bits的二维数组
        population = self.generate_initial_population()
        # print("population:",population)

        # 迭代num_generations轮
        best_fitness = float('-inf')
        best_individual = None
        best_fitnesses = []

        for generation in range(self.num_generations):
            # 二进制转换为十进制
            decimal_population = self.binary_to_decimal(population)
            # print("decimal_population:",decimal_population)
            # 计算适应度分数
            fitness_values = self.compute_fitness(decimal_population)
            # print("fitness_values:",fitness_values)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            merged_population = [individual[0] + individual[1] for individual in population]
            # 选择父代个体
            parents = self.selection(merged_population, fitness_values)
            # print("select parents:",parents)
            # 交叉操作
            offspring = self.crossover_with_avoidance(parents)
            # 变异操作
            after_offspring = self.mutation(offspring, -10, 10)

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
            best_individual_binary = self.decimal_to_binary(generation_best_individual)
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

        # 将最佳个体的十进制值转换为二进制编码并输出
        best_individual_decimal = best_individual
        best_individual_binary = self.decimal_to_binary(best_individual_decimal)
        flattened_best_individual_binary = list(chain.from_iterable(best_individual_binary))
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal}, Best individual (binary): {', '.join([str(x) for x in flattened_best_individual_binary])}")
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


class GeneticAlgorithmReal(GeneticAlgorithm):
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate, threshold, bounds, fitness_function_type):
        super().__init__(population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                         mutation_rate, threshold, bounds, fitness_function_type)

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            individual = []
            for j in range(self.num_variables):
                individual.append(np.random.uniform(low=self.bounds[j][0], high=self.bounds[j][1]))
            population.append(individual)
        return population

    def fitness_function(self, individual):
        x1, x2 = individual[0], individual[1]
        return self.fitness_function_instance.calculate(x1, x2)

    def euclidean_distance(self, u, v):
        u = np.array(u)
        v = np.array(v)
        dist = np.linalg.norm(u - v)  # 求向量差的范数，即欧式距离
        return dist

    # 带距离的交叉
    def crossover_with_avoidance(self, parents):
        offspring = []
        num_parents = len(parents)
        # print(parents)

        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            distance = self.euclidean_distance(parent1, parent2)
            threshold = 1
            if np.random.random() < self.crossover_rate and distance > threshold:
                alpha1 = np.random.uniform(0, 1)
                alpha2 = np.random.uniform(0, 1)
                offspring1 = [alpha1 * parent1[0] + (1 - alpha1) * parent2[0],
                              alpha2 * parent1[1] + (1 - alpha2) * parent2[1]]
                offspring2 = [alpha1 * parent2[0] + (1 - alpha1) * parent1[0],
                              alpha2 * parent2[1] + (1 - alpha2) * parent1[1]]

            else:
                offspring1 = parent1
                offspring2 = parent2

            offspring.append(offspring1)
            offspring.append(offspring2)

        return offspring

    def mutation(self, offspring, down, up):
        # 遍历每个后代
        for i in range(len(offspring)):
            # 遍历每个后代的基因
            for j in range(len(offspring[i])):
                # 判断是否进行变异操作
                if np.random.uniform(0, 1) <= self.mutation_rate:
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

    def run(self):
        # 初始化种群 生产的是population_size*num_bits的二维数组
        population = self.generate_initial_population()

        # 迭代num_generations轮
        best_fitness = float('-inf')
        best_individual = None
        best_fitnesses = []

        for generation in range(self.num_generations):
            # 计算适应度分数
            fitness_values = self.compute_fitness(population)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            # 选择父代个体
            parents = self.selection(population, fitness_values)
            # 交叉操作
            offspring = self.crossover_with_avoidance(parents)
            # 变异操作
            after_offspring = self.mutation(offspring, -10, 10)
            # 得到新的种群
            population = after_offspring
            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = fitness_values.index(max(fitness_values))
            # 计算新种群（经过选择、交叉和变异操作后的种群）的适应度分数
            new_fitness_values = self.compute_fitness(population)
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


class GeneticAlgorithm_varible_Binary(GeneticAlgorithm):
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate, threshold, bounds, fitness_function_type):
        super().__init__(population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                         mutation_rate, threshold, bounds, fitness_function_type)

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            individual = []
            for j in range(self.num_variables):
                lower_bound, upper_bound = self.num_bits[0], self.num_bits[1]
                num_bits = random.randint(lower_bound, upper_bound)
                individual.append([random.choice([0, 1]) for _ in range(num_bits)])
            population.append(individual)
        # print(population)
        return population

    # 从二进制编码转换为十进制数值
    def binary_to_decimal(self, population):
        decimal_population = []  # 存储所有染色体的十进制数值
        for individual in population:  # 遍历种群中的每个个体
            decimals = [[], []]  # 初始化包含两个变量的十进制数值列表
            for i, variable in enumerate(individual):  # 遍历个体中的每个变量
                decimal = 0  # 初始化十进制数值
                for j, gene in enumerate(variable):  # 遍历变量中的每个基因
                    decimal += gene * (2 ** j)  # 将基因的值乘以2的幂次方，求和得到十进制数值
                lower_bound, upper_bound = self.bounds[i]
                mapped_decimal = lower_bound + (decimal / ((2 ** len(variable)) - 1)) * (upper_bound - lower_bound)
                decimals[i].append(mapped_decimal)  # 将映射后的十进制数值添加到列表中
            decimal_population.append(decimals)  # 将包含两个变量的十进制数值列表添加到总列表中
        # print("decimal_population:",decimal_population)
        return decimal_population  # 返回所有染色体映射后的十进制数值列表

    # 十进制数值转换为二进制编码
    def decimal_to_binary(self, decimal_values):
        binary_values = []
        for i, decimal_value_list in enumerate(decimal_values):
            min_bound, max_bound = self.bounds[i]
            decimal_value = decimal_value_list[0]

            # 将十进制数值映射到整数范围
            fixed_point_value = int((decimal_value - min_bound) / (max_bound - min_bound) * (2 ** self.num_bits - 1))

            # 将整数转换为二进制编码列表
            # format() 函数将整数转换为二进制字符串，并指定了字符串的位数
            binary_value = [int(bit) for bit in format(fixed_point_value, f'0{self.num_bits}b')]
            binary_values.append(binary_value)

        return binary_values

    # 计算汉明码距离
    def hamming_distance(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    # 带距离的交叉
    def crossover_with_avoidance(self, parents):
        offspring = []
        num_parents = len(parents)

        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            distance = self.hamming_distance(parent1, parent2)
            # print(distance)
            if np.random.random() < self.crossover_rate and distance > self.threshold:
                crossover_point = np.random.randint(1, len(parent1))
                offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
                offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                offspring1 = parent1
                offspring2 = parent2

            offspring.append(offspring1)
            offspring.append(offspring2)

        return offspring

    def mutation(self, offspring, down, up):
        # 遍历每个后代
        for i in range(len(offspring)):
            # 遍历每个后代的基因
            for j in range(len(offspring[i])):
                # 判断是否进行变异操作
                if np.random.uniform(0, 1) <= self.mutation_rate:
                    # 二进制编码：随机将基因进行变异
                    offspring[i][j] = 1 - offspring[i][j]
        # 返回变异后的后代
        return offspring

    def run(self):
        # 初始化种群 生产的是population_size*num_bits的二维数组
        population = self.generate_initial_population()
        # print("population:",population)

        # 迭代num_generations轮
        best_fitness = float('-inf')
        best_individual = None
        best_fitnesses = []

        for generation in range(self.num_generations):
            # 二进制转换为十进制
            decimal_population = self.binary_to_decimal(population)
            # print("decimal_population:",decimal_population)
            # 计算适应度分数
            fitness_values = self.compute_fitness(decimal_population)
            # print("fitness_values:",fitness_values)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            merged_population = [individual[0] + individual[1] for individual in population]
            # 选择父代个体
            parents = self.selection(merged_population, fitness_values)
            # print("select parents:",parents)
            # 交叉操作
            offspring = self.crossover_with_avoidance(parents)
            # 变异操作
            after_offspring = self.mutation(offspring, -10, 10)

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
            # best_individual_binary = self.decimal_to_binary(generation_best_individual)
            # print(best_individual_binary)
            flattened_individual = list(chain.from_iterable(generation_best_individual))
            print(
                f"Generation {generation + 1} - Best fitness: {generation_best_fitness:.6f}, Decimal: {', '.join([f'{x:.6f}' for x in flattened_individual])}")
            # 更新全局最优解
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best_individual

                # # 如果找到了Best fitness大于 36300，就退出循环
                # if generation_best_fitness >= 36300:
                #     print(f"Solution found after {generation + 1} generations.")
                #     break

        # 将最佳个体的十进制值转换为二进制编码并输出
        best_individual_decimal = best_individual
        # best_individual_binary = self.decimal_to_binary(best_individual_decimal)
        # flattened_best_individual_binary = list(chain.from_iterable(best_individual_binary))
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal}")
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


class GeneticAlgorithm_VariableLength(GeneticAlgorithm_varible_Binary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def crossover_with_avoidance(self, parents):
        offspring = []
        num_parents = len(parents)

        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            distance = self.hamming_distance(parent1, parent2)

            if random.random() < self.crossover_rate and distance > self.threshold:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)))
                offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
                offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                offspring1 = parent1
                offspring2 = parent2

            offspring.append(offspring1)
            offspring.append(offspring2)

        return offspring

    def mutation(self, offspring, down, up):
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if random.uniform(0, 1) <= self.mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

            # 随机增加或删除基因
            if random.uniform(0, 1) <= self.mutation_rate:
                if random.choice([True, False]):
                    # 添加基因
                    offspring[i].append(random.choice([0, 1]))
                else:
                    # 删除基因
                    if len(offspring[i]) > 1:
                        del offspring[i][-1]

        return offspring


class GeneticAlgorithmBinaryWithGradient(GeneticAlgorithmBinary):
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate, threshold, bounds, fitness_function_type, local_search_rate=0.2, gradient_learning_rate=0.1, gradient_max_iter=50):
        super().__init__(population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                         mutation_rate, threshold, bounds, fitness_function_type)
        self.local_search_rate = local_search_rate
        self.gradient_learning_rate = gradient_learning_rate
        self.gradient_max_iter = gradient_max_iter

    # 梯度下降函数
    def gradient_descent(self, individual_decimal):
        x1, x2 = individual_decimal[0][0], individual_decimal[1][0]
        for _ in range(self.gradient_max_iter):
            gradient_x1 = 12 * x1 * (x1 ** 2 - x2)
            gradient_x2 = -6 * (x1 ** 2 - x2)

            # 防止溢出
            if abs(gradient_x1) > 10 or abs(gradient_x2) > 10:
                break

            x1 = x1 - self.gradient_learning_rate * gradient_x1
            x2 = x2 - self.gradient_learning_rate * gradient_x2
        return [[x1], [x2]]

    # 添加局部搜索方法
    def apply_local_search(self, decimal_population):
        for i in range(len(decimal_population)):
            if np.random.random() < self.local_search_rate:
                decimal_population[i] = self.gradient_descent(decimal_population[i])
        return decimal_population

    # 重写遗传算法的运行方法
    def run(self):
        # 初始化种群 生产的是population_size*num_bits的二维数组
        population = self.generate_initial_population()
        # print("population:",population)

        # 迭代num_generations轮
        best_fitness = float('-inf')
        best_individual = None
        best_fitnesses = []

        for generation in range(self.num_generations):
            # 二进制转换为十进制
            decimal_population = self.binary_to_decimal(population)

            # 应用局部搜索
            decimal_population = self.apply_local_search(decimal_population)

            # 计算适应度分数
            fitness_values = self.compute_fitness(decimal_population)
            # print("fitness_values:",fitness_values)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            merged_population = [individual[0] + individual[1] for individual in population]
            # 选择父代个体
            parents = self.selection(merged_population, fitness_values)
            # print("select parents:",parents)
            # 交叉操作
            offspring = self.crossover_with_avoidance(parents)
            # 变异操作
            after_offspring = self.mutation(offspring, -10, 10)

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
            best_individual_binary = self.decimal_to_binary(generation_best_individual)
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

            # 将最佳个体的十进制值转换为二进制编码并输出
        best_individual_decimal = best_individual
        best_individual_binary = self.decimal_to_binary(best_individual_decimal)
        flattened_best_individual_binary = list(chain.from_iterable(best_individual_binary))
        print(
            f"\nFinal result - Best fitness: {best_fitness:.6f}, Best individual (decimal): {best_individual_decimal}, Best individual (binary): {', '.join([str(x) for x in flattened_best_individual_binary])}")
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


class GeneticAlgorithmRealWithGradient(GeneticAlgorithmReal):
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                 mutation_rate, threshold, bounds, fitness_function_type, local_search_rate=0.2, gradient_learning_rate=0.1, gradient_max_iter=50):
        super().__init__(population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate,
                         mutation_rate, threshold, bounds, fitness_function_type)
        self.local_search_rate = local_search_rate
        self.gradient_learning_rate = gradient_learning_rate
        self.gradient_max_iter = gradient_max_iter

    def gradient_descent(self, individual):
        x1, x2 = individual[0], individual[1]
        for _ in range(self.gradient_max_iter):
            gradient_x1 = 12 * x1 * (x1 ** 2 - x2)
            gradient_x2 = -6 * (x1 ** 2 - x2)

            # 防止溢出
            if abs(gradient_x1) > 10 or abs(gradient_x2) > 10:
                break

            x1 = x1 - self.gradient_learning_rate * gradient_x1
            x2 = x2 - self.gradient_learning_rate * gradient_x2
        return [x1, x2]

    def apply_local_search(self, population):
        for i in range(len(population)):
            if np.random.random() < self.local_search_rate:
                population[i] = self.gradient_descent(population[i])
        return population

    def run(self):
        # 初始化种群 生产的是population_size*num_bits的二维数组
        population = self.generate_initial_population()

        # 迭代num_generations轮
        best_fitness = float('-inf')
        best_individual = None
        best_fitnesses = []

        for generation in range(self.num_generations):
            population = self.apply_local_search(population)
            # 计算适应度分数
            fitness_values = self.compute_fitness(population)
            # 将当前种群深度拷贝一份用于下一代操作，避免直接修改当前种群
            next_generation = copy.deepcopy(population)
            # 选择父代个体
            parents = self.selection(population, fitness_values)
            # 交叉操作
            offspring = self.crossover_with_avoidance(parents)
            # 变异操作
            after_offspring = self.mutation(offspring, -10, 10)
            # 得到新的种群
            population = after_offspring
            # 找到当前一代中的最大适应度值的下标
            max_fitness_index = fitness_values.index(max(fitness_values))
            # 计算新种群（经过选择、交叉和变异操作后的种群）的适应度分数
            new_fitness_values = self.compute_fitness(population)
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
