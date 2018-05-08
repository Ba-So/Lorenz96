#!/usr/bin/env python
# coding=utf-8
import lorenz as lz

if __name__ == '__main__':

    lz_irr = lz.L96(10,8)
    lz_irr.Integrate()
    lz_irr.Plot()
