#!/usr/bin/env python
# coding=utf-8

import numpy as np
# ugly workaround
import PyQt4
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

class L96:
    def __init__(self, N, R, t = None):
        self.N = N
        self.R = R
        self.F = np.ones(N) * R
        if t == None:
            t = np.arange(0.0, 30.0, 0.01)
        self.t = t
        self.y = np.zeros((N, len(self.t), N))

    def Lorenz(self, y, t):
        dy = np.zeros(self.N)
        dy[0] = ((y[1] - y[self.N-2])
                  * y[self.N-1] - y[0])
        dy[1] = ((y[2] - y[self.N-1])
                  * y[0] - y[1])
        dy[self.N-1] = ((y[0]-y[self.N-3])
                    * y[self.N-2] - y[self.N-1])
        for j in range(2, (self.N-1)):
            dy[j] = (
                (y[j+1] - y[j-2])
                * y[j-1] - y[j]
            )
        dy = dy + self.F[self.j]
        return dy

    def Integrate(self):
        self.j = 0
        while self.j < len(self.F):
            ci = np.random.rand(self.N)
            self.y[self.j, :, :] = odeint(
                self.Lorenz, ci, self.t,
                rtol = 1e-8, atol = 1e-8)
            self.j = self.j + 1

    def Momentum(self):
        self.Mom = np.einsum('ijk -> ik',self.y) / self.N
        self.meanMom = np.mean(self.Mom, 1)

    def Energy(self):
        self.Egy = np.einsum('ijk, ijk -> ik', self.y, self.y) / (self.N * 2)
        self.meanEgy = np.mean(self.Egy, 1)

    def Plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.y[1, :, 0], self.y[1, :, 1],  self.y[1, :, 2])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        plt.show()






