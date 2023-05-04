# -*- coding: utf-8 -*-
# @Time : 2023/4/20 15:37
# @Author : Ricardo_PING
# @File : GeneticAlgorithmFactory
# @Project : Genetic Algorithm.py
# @Function ：

# GeneticAlgorithmFactory.py
from Problem import SampleProblem, Init_var, Oper_var, BinaryWithGradient, RealWithGradient


class GeneticAlgorithmFactory:
    @staticmethod
    def create_genetic_algorithm(problem_type, *args, **kwargs):
        if problem_type == 'SampleProblem':
            return SampleProblem(*args, **kwargs)
        # Add other problem types here
        elif problem_type == 'Init_var':
            return Init_var(*args, **kwargs)
        elif problem_type == 'Oper_var':
            return Oper_var(*args, **kwargs)
        elif problem_type == 'BinaryWithGradient':
            return BinaryWithGradient(*args, **kwargs)
        elif problem_type == 'RealWithGradient':
            return RealWithGradient(*args, **kwargs)
        else:
            raise ValueError("Invalid problem type")
