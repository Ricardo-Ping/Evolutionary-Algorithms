# -*- coding: utf-8 -*-
# @Time : 2023/4/20 15:36
# @Author : Ricardo_PING
# @File : Problem
# @Project : Genetic Algorithm.py
# @Function ：

# Problem.py
from GeneticAlgorithm import GeneticAlgorithm, GeneticAlgorithmBinary, GeneticAlgorithmReal


class SampleProblem:
    def __init__(self, population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate, mutation_rate,
                 threshold, flag, bounds, fitness_function_type):
        if flag == "binary":
            self.ga_instance = GeneticAlgorithmBinary(
                population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate, mutation_rate,
                threshold, bounds, fitness_function_type)
        elif flag == "real":
            self.ga_instance = GeneticAlgorithmReal(
                population_size, num_variables, num_bits, num_generations, num_parents, crossover_rate, mutation_rate,
                threshold, bounds, fitness_function_type)
        else:
            raise ValueError("Invalid flag")

    def run(self):
        self.ga_instance.run()

    # Override/extend any methods specific to the problem

