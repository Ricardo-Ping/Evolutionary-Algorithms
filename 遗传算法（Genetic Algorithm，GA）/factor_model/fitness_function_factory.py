# -*- coding: utf-8 -*-
# @Time : 2023/4/20 16:04
# @Author : Ricardo_PING
# @File : fitness_function_factory
# @Project : Genetic Algorithm.py
# @Function ：
# fitness_function_factory.py
from fitness_function import MyFitnessFunction, Bent_Cigar


class FitnessFunctionFactory:

    @staticmethod
    def create_fitness_function(function_type):
        if function_type == 'my_fitness_function':
            return MyFitnessFunction()
        elif function_type == 'Bent_Cigar':
            return Bent_Cigar()
        else:
            raise ValueError("Invalid fitness function type")
