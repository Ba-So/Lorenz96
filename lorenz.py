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

class L96(object):
    def __init__(self, N, R, t = None):
        self.N = N
        self.R = R
        self.F = np.ones(N) * R
        if t == None:
            t = np.arange(0.0, 30.0, 0.01)
        self.t = t
        self.y = np.zeros((N, len(self.t), N))


    def Integrate(self):
        self.j = 0
        while self.j < len(self.F):
            ci = np.random.rand(self.N)
            self.y[self.j, :, :] = odeint(
                self.Lorenz, ci, self.t,
                rtol = 1e-8, atol = 1e-8)
            self.j = self.j + 1

    def Lorenz(self, y, t):
        #tbd in child classes

        return y

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

class L96_irr(L96):

    def __init__(self, N, R, t = None):
        super(L96_irr, self).__init__(N, R, t)
        self.J = np.array([
            [ for k in range(N)]
            for j in range(N)
        ])

    def Lorenz(self, y, t):
        dy = np.zeros(self.N*(self.N+1))
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

        # the Jacobian is riding piggy back...
        dy[self.N:self.N*self.N] = self.Jacobian(y)

        return dy

    def Jacobian(self, y):

        Y = np.empty([self.N, self.N])
        Y.fill(0.0)
        for j in range(self.N):
            for k in range(self.N):
                Y[j,k] = y[self.N-1 + j + (k-1)*(self.N-1)]

        Jac1 = np.empty([self.N, self.N])
        Jac1.fill(0.0)
        Jac1 = np.fill_diagonal(Jac1, -1)
        Jac2 = np.empty([self.N, self.N])
        Jac2.fill(0.0)
        Jac2[0, 1] = y[self.N-1]
        Jac2[0, self.N - 1] = y[1]
        Jac2[0, self.N - 2] = -y[self.N-1]

        Jac2[1, 2] = y[0]
        Jac2[1, self.N - 1] = -y[0]
        Jac2[1, 0] = y[2] - y[self.N-1]

        Jac2[self.N-1, 0] = y[self.N-2]
        Jac2[self.N-1, self.N - 3] = -y[self.N-2]
        Jac2[self.N-1, self.N - 2] = y[0] - y[self.N-3]

        for j in range(2,self.N-1):
            Jac2[j, j+1] = y[j-1]
            Jac2[j, j-1] = y[j+1]-y[j-2]]
            Jac2[j, j-2] = -y[j-1]

        Jac = Jac1 + Jac2

        # expected shape of that product is 1D len(self.N*(self.N-1))
        return Jac * Y


class L96_rev(L96):

    def __init__(self, N, R, t = None):
        super(L96_rev, self).__init__(N, R, t)


    def Lorenz(self, y, t):
        Fg2 = (
            np.ones(self.N)
            * np.divide(
                np.einsum('j -> ',y),
                np.einsum('j, j -> ', y,y)
              )
       )
        dy = np.zeros(self.N)
        dy[0] = (
            (y[1] - y[self.N-2]) * y[self.N-1]
            - Fg2[0]*y[0]
            + self.F[0]
        )
        dy[1] = (
            (y[2] - y[self.N-1]) * y[0]
            - Fg2[1] * y[1]
            + self.F[1]
        )
        dy[self.N-1] = (
            (y[0]-y[self.N-3]) * y[self.N-2]
            - Fg2[self.N - 1] * y[self.N-1]
            + self.F[self.N -1]
        )
        for j in range(2, (self.N-1)):
            dy[j] = (
                (y[j+1] - y[j-2]) * y[j-1]
                - Fg2[j]*y[j]
                + self.F[j]
            )
        dy = dy + self.F[self.j]
        return dy




