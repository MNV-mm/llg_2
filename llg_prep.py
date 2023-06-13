#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:53:22 2022

@author: mnv
"""
import DD_Hd
import funcs_2
#import dolfin as dp

Lx = 570 # 60 150 80
Ly = 200 # 30 80 40

route_0 = '/home/mnv/llg_nl/'

#hd_ext_expr = funcs_2.n_pair(Ly, 30, 10, 9, 3)
#hd = dp.Expression(hd_ext_expr, degree = 3)

DD_Hd.pe_EF(45,200,1,Lx,Ly,15, route_0 + 'results/e_field/')
