# -*- coding: utf-8 -*-
# @Time : 2023/4/20 15:37
# @Author : Ricardo_PING
# @File : GeneticAlgorithmFactory
# @Project : Genetic Algorithm.py
# @Function ：

# GeneticAlgorithmFactory.py
from Problem import SampleProblem, Bent_Cigar


class GeneticAlgorithmFactory:
    @staticmethod
    def create_genetic_algorithm(problem_type, *args, **kwargs):
        if problem_type == 'SampleProblem':
            return SampleProblem(*args, **kwargs)
        # Add other problem types here
        elif problem_type == 'Bent_Cigar':
            return Bent_Cigar(*args, **kwargs)
        else:
            raise ValueError("Invalid problem type")
