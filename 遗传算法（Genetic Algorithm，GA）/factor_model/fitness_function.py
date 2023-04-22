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


class MyFitnessFunction(FitnessFunction):

    def calculate(self, x1, x2):
        y = 3 * (x1 ** 2 - x2) ** 2
        return y

