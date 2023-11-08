# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:22:08 2023

@author: andja
"""

# Importation des modules
# import time

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import sys
import pytest



# sys.path.append('../Fonctions')

try:
    from projet_fct_nesquik import *
except:
    pass

# k= ij2k(5,4, 5)
# print(k)

# i, j = k2ij(29, 5)
# print(i,j)

print(k2ij_matrix(5, range(0,35)))