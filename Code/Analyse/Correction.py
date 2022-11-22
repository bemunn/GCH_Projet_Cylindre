# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:21:29 2022

@author: guaugg
"""

# Importation des modules
# import time

import numpy as np
import pandas as pd
import sys


sys.path.append('../Fonctions')

try:
    from projet_fct import *
except:
    print("ERREUR! Il y a une erreur fatale dans le fichier projet_fct.py")

class parametres():
    U_inf = 1       # Vitesse du fluide éloigné du cylindre [-]
    R     = 1       # Rayon interne du cylindre creux [-]
    R_ext = 5       # Rayon externe du cylindre creux [-]   

prm = parametres()

class Test:

    def test_positions(self):
            
        X = [-1,1]
        Y = [0,2]
        nx = 3
        ny = 4
        x,y = position(X,Y,nx,ny)
        assert(np.asarray(x).shape == (ny,nx))
        assert(np.asarray(y).shape == (ny,nx))
        assert(all(abs(np.asarray(x[0,:]) - np.array([-1,0,1])) < 1e-03))
        assert(all(abs(np.asarray(y[:,0]) - np.array([2,1.3333,0.6666,0])) < 1e-03))

    def test_assemblage(self):
        nr = 51
        ntheta = 76
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

        psi_num = mdf_assemblage(nr,ntheta, prm)
        psi_exact = np.zeros(nr*ntheta)
        
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
        assert(len(psi_num) == nr * ntheta)
        assert(all(abs(psi_num - psi_exact) < 1e-03))
        
    def test_vitesse_polaire(self):
        nr = 201
        ntheta = 221
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        
        psi_num = mdf_assemblage(nr,ntheta, prm)
        vr, vtheta = vitesse_polaire(psi_num,nr,ntheta, prm )

        vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
        vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)
        
        assert(np.asarray(vr).shape == (ntheta,nr))
        assert(np.asarray(vtheta).shape == (ntheta,nr))
        
        for i in range(0,ntheta):
            assert(all(abs(vr[i,:] - vr_exact[i,:]) < 1e-03))
            assert(all(abs(vtheta[i,:] - vtheta_exact[i,:]) < 1e-03))
            