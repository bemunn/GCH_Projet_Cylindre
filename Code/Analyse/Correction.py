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

        psi_num = mdf(nr,ntheta, prm)
        psi_exact = np.zeros(nr*ntheta)
        
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
        assert(len(psi_num) == nr * ntheta)
        assert(all(abs(psi_num - psi_exact) < 1e-03))
        
    def test_vitesse_polaire(self):
        nr = 201
        ntheta = 201
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        
        psi_exact = np.zeros(nr*ntheta)
        
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
        vr, vtheta = vitesse_polaire(psi_exact,nr,ntheta, prm )

        vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
        vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)
        
        assert(np.asarray(vr).shape == (ntheta,nr))
        assert(np.asarray(vtheta).shape == (ntheta,nr))
        
        for i in range(0,ntheta):
            assert(all(abs(vr[i,:] - vr_exact[i,:]) < 1e-03))
            assert(all(abs(vtheta[i,:] - vtheta_exact[i,:]) < 1e-03))
            
    def test_polaire2xy(self):
        nr = 51
        ntheta = 76
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        psi_exact = np.zeros(nr*ntheta)
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
                
        v_r, v_theta = vitesse_polaire(psi_exact,nr,ntheta, prm )
        v_x_exact = np.zeros([ntheta, nr])
        v_y_exact = np.zeros([ntheta, nr])

        for i in range(nr):            
            for j in range(ntheta):
                    v_x_exact[-j-1, i] = v_r[-j-1, i] * np.cos(theta[-1-j,i]) + v_theta[-j-1,i] * np.cos(theta[-1-j,i] + np.pi/2)
                    print(i)
                    v_y_exact[-j-1, i] = v_r[-j-1, i] * np.sin(theta[-1-j,i]) + v_theta[-j-1,i] * np.sin(theta[-1-j,i] + np.pi/2)
                    
        vx, vy = polaire2xy(v_r, v_theta, nr,ntheta, prm)
        
        assert(np.asarray(vx).shape == (ntheta,nr))
        assert(np.asarray(vy).shape == (ntheta,nr))
        for i in range(0,ntheta):
            assert(all(abs(vx[i,:] - v_x_exact[i,:]) < 1e-03))
            assert(all(abs(vy[i,:] - v_y_exact[i,:]) < 1e-03))
            
    def test_coeff_pression(self):
        nr = 411
        ntheta = 411
        r_mat,theta_mat = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        theta = theta_mat[:,0]
        r = prm.R
        psi_exact = np.zeros(nr*ntheta)
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r_mat[-1-j,i] * np.sin(theta_mat[-1-j,i]) * (1 - prm.R**2 / r_mat[-1-j,i]**2)   
        v_r, v_theta = vitesse_polaire(psi_exact,nr,ntheta, prm)
         
        cp_exact = -1 + 2 * (np.cos(theta)**2 - np.sin(theta)**2)
        
        cp = coeff_pression(v_r[:,0], v_theta[:,0], prm)
        
        assert(len(cp) == ntheta)
        for i in range(0,ntheta):
            assert(all(abs(cp - cp_exact) < 1e-03))
   
    def test_trapeze(self):
        v1 = np.array([0,2,4,5,6,10])
        v2 = np.array([0,1,3,7,15,18,20])
        t1 = np.linspace(0,(len(v1)-1)*2,len(v1))
        t2 = np.linspace(0,(len(v2)-1)*3,len(v2))
        pos1 = np.array([0,2,8,17,28,44])
        pos2 = np.array([0.,1.5,7.5,22.5,55.5,105.,162.])
        
        rep_trapeze1 = trapeze(t1,v1)
        rep_trapeze2 = trapeze(t2,v2)
        
        assert(abs(rep_trapeze1 - pos1[-1]) < 1e-02)
        assert(abs(rep_trapeze2 - pos2[-1]) < 1e-02)          
   
    def test_coeff_aerodynamique(self):
        nr = 411
        ntheta = 411
        r_mat,theta_mat = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        theta = theta_mat[:,0]
        r = prm.R
        psi_exact = np.zeros(nr*ntheta)
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r_mat[-1-j,i] * np.sin(theta_mat[-1-j,i]) * (1 - prm.R**2 / r_mat[-1-j,i]**2) 
                
        v_r_mat, v_theta_mat = vitesse_polaire(psi_exact,nr,ntheta, prm)
        cd, cl = coeff_aerodynamique(v_r_mat, v_theta_mat, nr, ntheta, prm)
        cd_exact = 0
        cl_exact = 0
        assert(abs(cd - cd_exact) < 1e-05)
        assert(abs(cl - cl_exact) < 1e-05)
        
  
        
         
         
    