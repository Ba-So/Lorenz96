#!/usr/bin/env python
# coding=utf-8
import lorenz as lz

if __name__ == '__main__':

    lz_irr = lz.L96_irr(32,8)
    lz_irr.Integrate()
    lz_irr.Plot()
    lz_irr.Momentum()
    lz_irr.Energy()
    print(lz_irr.meanEgy)
    print(lz_irr.meanMom)

    lz_rev = lz.L96_rev(32,8)
    lz_rev.Integrate()
    lz_rev.Plot()
    lz_rev.Momentum()
    lz_rev.Energy()
    print(lz_rev.meanEgy)
    print(lz_rev.meanMom)
