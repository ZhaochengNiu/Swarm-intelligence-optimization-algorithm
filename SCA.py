# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/8 14:20
# @Author : LucXiong
# @Project : Model
# @File : SCA_new.py

"""
Ref:https://github.com/luizaes/sca-algorithm
S. Mirjalili, SCA: A Sine Cosine Algorithm for Solving Optimization Problems, Knowledge-based Systems, in press, 2015, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import test_function


class sca():
    def __init__(self, pop_size=5, n_dim=2, a=2, lb=-1e5, ub=1e5, max_iter=20, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.a = a # 感知概率
        self.func = func
        self.max_iter = max_iter  # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))] # y = f(x) for all particles
        # X[i] 表示 X 的第 i 行
        # len(X) 表示 X 的行数
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_x = self.pbest_x.mean(axis=0)
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy() # copy很重要！
            self.gbest_y = self.pbest_y[idx_min]

    def update(self, i):
        r1 = self.a - i * (self.a / self.max_iter)
        for j in range(self.pop):
            for k in range(self.n_dim):
                r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                r3 = 2 * random.uniform(0.0, 1.0)
                r4 = random.uniform(0.0, 1.0)
                if r4 < 0.5:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
                else:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # Function for fitness evaluation of new solutions

    def run(self):
        for i in range(self.max_iter):
            self.update(i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        best_x, best_y = self.gbest_x, self.gbest_y
        return best_x, best_y


if __name__ == '__main__':
    n_dim = 2
    lb = [-512 for i in range(n_dim)]
    ub = [512 for i in range(n_dim)]
    demo_func = test_function.f23
    sca = sca(n_dim=2, pop_size=40, max_iter=150, lb=lb, ub=ub, func=demo_func)
    sca.run()
    print('best_x is ', sca.gbest_x, 'best_y is', sca.gbest_y)
    print(f'{demo_func(sca.gbest_x)}\t{sca.gbest_x}')
    plt.plot(sca.gbest_y_hist)
    plt.show()

    # # test 1
    # print(np.array(3.5) * np.ones(5))
    # LB = [-5, 0]
    # HB = [0, 5]
    # LB, HB = np.array(LB) * np.ones(2), np.array(HB) * np.ones(2)
    # print(LB)
    # print(HB)
    # X = np.random.uniform(low=LB, high=HB, size=(10, 2))
    # gbest_x1 = X.mean(axis=0).reshape(1, -1)
    # print(gbest_x1)
    # gbest_x1 = X.mean(axis=0)
    # print(gbest_x1)
    # gbest_x2 = X[0, :].copy()
    # print(gbest_x2)
    # print(X[1]) # 输出 X 的第一行
    # print(len(X)) # 输出 X 的行数
    # print(X)
    # LClip = [-1, 0]
    # HClip = [0, 1]
    # X = np.clip(X, LClip, HClip)
    # print(X)

    # # test 2
    # x = [1, 2]
    # y = [0, 0]
    # y[0] = x[0]
    # x[0] = 10
    # print(y)

    # # test 2
    # x = [1, 2]
    # y = x
    # x[0] = 10
    # print(y)

    # pbest_y = [0,1,2,3]
    # gbest_y = pbest_y[1]
    # pbest_y[1] = 10
    # print(gbest_y)

    # pbest_y = [i for i in range(0, 10)]
    # print(pbest_y)
    # idx_min = pbest_y.index(min(pbest_y))
    # print(idx_min)

    # X = np.random.uniform(low=0, high=5, size=(10, 2))
    # gbest_x = X[0, :]
    # X[0, 0] = 0
    # X[0, 1] = 0
    # print(gbest_x)

    # x = np.ones((4,4))
    # y = x.copy()
    # y[1] = x[1]
    # x[1] = [10,10,10,10]
    # print(y)
###################################################
    # 1、非数组变量
    # 一般的在python中我们将变量a的值赋值给变量b，可以进行如下操作
    # a = 1
    # b = a
    # b += 1
    # print( "a =" , a )
    # print( "b =" , b )
    # 从结果中可以看出，我改变b的值，并不会影响a。也就是说对于非数组、列表、字典等类型的变量，直接进行复制，变量b保存的不是地址。
    # 2、矩阵
    # 2.1 使用向量给向量进行赋值
    # 对向量进行赋值操作：
    # x = np.mat( '1 2 3' )
    # y = x
    # y[0] += 1
    # print("x = ", x)
    # print("y = ", y)
    # 可以看出，改变y的第一个元素的值，x中对应元素值也随之改变，这说明这里保存的是地址。
    # 2.2
    # 使用矩阵给矩阵进行赋值
    # 使用矩阵对矩阵进行赋值：
    # x = np.mat('1 2 3 ; 4 5 6')
    # y = x
    # y[0, 0] += 1
    # print("x = \n", x)
    # print("y = \n", y)
    # 2.3
    # 使用矩阵中的某一行（或某一列）赋值给一个向量
    # x = np.mat('1 2 3 ; 4 5 6')
    # y = x[0, :]
    # y[0, 0] += 1
    # print("x = \n", x)
    # print("y = \n", y)
    # 从结果可以看出，x中的第一行元素和y中元素地址是相同的。