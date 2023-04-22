# -*- coding: utf-8 -*-
# @Time : 2023/4/20 15:37
# @Author : Ricardo_PING
# @File : main
# @Project : Genetic Algorithm.py
# @Function ：

# main.py
# main.py
from GeneticAlgorithmFactory import GeneticAlgorithmFactory

# 设置遗传算法的参数
population_size = 100
num_variables = 2
num_generations = 100
num_parents = 20
num_bits = 10  # 每个个体包含的基因数量
crossover_rate = 0.8
mutation_rate = 0.05
threshold = 4
flag = "real"  # "binary"或 "real"
bounds = [(-10, 10), (-10, 10)]  # 设置变量的范围

if __name__ == '__main__':
    # 使用 GeneticAlgorithmFactory 创建 SampleProblem 实例
    sample_problem = GeneticAlgorithmFactory.create_genetic_algorithm(
        problem_type='SampleProblem',
        population_size=population_size,
        num_variables=num_variables,
        num_bits = num_bits,
        num_generations=num_generations,
        num_parents=num_parents,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        threshold=threshold,
        flag=flag,
        bounds=bounds,
        fitness_function_type='my_fitness_function'
    )
    # 运行遗传算法
    sample_problem.run()
