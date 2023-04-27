# -*- coding: utf-8 -*-
# @Time : 2023/4/20 16:03
# @Author : Ricardo_PING
# @File : fitness_function
# @Project : Genetic Algorithm.py
# @Function ：
# fitness_function.py
from abc import ABC, abstractmethod


class FitnessFunction(ABC):

    @abstractmethod
    def calculate(self, x1, x2):
        pass

    @abstractmethod
    def calculate_3(self, x1, x2, x3):
        pass

    @abstractmethod
    def calculate_4(self, x1, x2, x3, x4):
        pass


class MyFitnessFunction(FitnessFunction):

    def calculate_4(self, x1, x2, x3, x4):
        pass

    def calculate_3(self, x1, x2, x3):
        pass

    def calculate(self, x1, x2):
        y = 3 * (x1 ** 2 - x2) ** 2
        return y


class Bent_Cigar(FitnessFunction):

    def calculate_4(self, x1, x2, x3, x4):
        pass

    def calculate_3(self, x1, x2, x3):
        pass

    def calculate(self, *args):
        x1 = args[0]
        xs = args[1:]
        y = x1 ** 2 + 1e6 * sum(map(lambda x: x ** 2, xs))
        return y
