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

#définition des paramètres utilisés dans le projet
class parametres():
    U_inf = 1       # Vitesse du fluide éloigné du cylindre [-]
    R     = 1       # Rayon interne du cylindre creux [-]
    R_ext = 5       # Rayon externe du cylindre creux [-]   

prm = parametres()

class Test:


    #test de la fonction mdf et vérification des résultats de la fonction de courrant
    def test_mdf(self):
        #nombre de points utilisés
        nr = 51
        ntheta = 76
        #discrétisation des r et theta
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        
        #résultat numérique
        psi_num = mdf(nr,ntheta, prm)
        
        #initialisation des résultats analytiques
        psi_exact = np.zeros(nr*ntheta)
        
        #boucles pour obtenir résultats analytiques
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
                
        #vérification dimensions vecteur numérique
        assert(len(psi_num) == nr * ntheta)
        #vérification des valeurs du vecteur numériques
        assert(all(abs(psi_num - psi_exact) < 1e-03))
        
    #test de la fonction vitesse_polaire et vérification des résultats numériques des vitesses polaires
    def test_vitesse_polaire(self):
        #nombre de points utilisés
        nr = 201
        ntheta = 201
        #discrétisation des r et theta       
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        
        #initialisation des résultats analytiques de la fonction de courrant
        psi_exact = np.zeros(nr*ntheta)
        
        #boucles pour obtenir résultats analytiques de la fonction de courrant
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
                
        #résultat numérique
        vr, vtheta = vitesse_polaire(psi_exact,nr,ntheta, prm )

        #résultats analytiques
        vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
        vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)
        
        #vérification des dimensions des matrices numériques
        assert(np.asarray(vr).shape == (ntheta,nr))
        assert(np.asarray(vtheta).shape == (ntheta,nr))
        
        #vérification des valeurs des matrices numériques
        for i in range(0,ntheta):
            assert(all(abs(vr[i,:] - vr_exact[i,:]) < 1e-03))
            assert(all(abs(vtheta[i,:] - vtheta_exact[i,:]) < 1e-03))
         
    #test de la fonction polaire2xy (conversion des vitesses polaires en cartésienne)
    def test_polaire2xy(self):
        #nombre de points utilisés
        nr = 51
        ntheta = 76
        #discrétisation des r et theta       
        r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        
        #initialisation des résultats analytiques de la fonction de courrant
        psi_exact = np.zeros(nr*ntheta)
        
        #boucles pour obtenir résultats analytiques de la fonction de courrant
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
        
        #obtention des v_r et v_theta
        v_r, v_theta = vitesse_polaire(psi_exact,nr,ntheta, prm )
        
        #initialisation des matrices des vitesses cartésiennes analytiques
        v_x_exact = np.zeros([ntheta, nr])
        v_y_exact = np.zeros([ntheta, nr])

        #boucles pour obtenir les matrices des vitesses cartésiennes analytiques
        for i in range(nr):            
            for j in range(ntheta):
                    v_x_exact[-j-1, i] = v_r[-j-1, i] * np.cos(theta[-1-j,i]) + v_theta[-j-1,i] * np.cos(theta[-1-j,i] + np.pi/2)
                    print(i)
                    v_y_exact[-j-1, i] = v_r[-j-1, i] * np.sin(theta[-1-j,i]) + v_theta[-j-1,i] * np.sin(theta[-1-j,i] + np.pi/2)
        
        #obtention des vitesses vx et vy par la fonction polaire2xy
        vx, vy = polaire2xy(v_r, v_theta, nr,ntheta, prm)
        
        #vérification des dimensions des matrices numériques
        assert(np.asarray(vx).shape == (ntheta,nr))
        assert(np.asarray(vy).shape == (ntheta,nr))
        
        #vérification des valeurs des matrices numériques
        for i in range(0,ntheta):
            assert(all(abs(vx[i,:] - v_x_exact[i,:]) < 1e-03))
            assert(all(abs(vy[i,:] - v_y_exact[i,:]) < 1e-03))
            
    
    #test de la fonction coeff_pression et vérification des résultats numériques du coefficient de pression
    def test_coeff_pression(self):
        #nombre de points utilisés
        nr = 411
        ntheta = 411
        
        #discrétisation des r et theta       
        r_mat,theta_mat = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        theta = theta_mat[:,0]
        r = prm.R
        
        #initialisation des résultats analytiques de la fonction de courrant
        psi_exact = np.zeros(nr*ntheta)
        
        #boucles pour obtenir résultats analytiques de la fonction de courrant
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r_mat[-1-j,i] * np.sin(theta_mat[-1-j,i]) * (1 - prm.R**2 / r_mat[-1-j,i]**2) 
                
        #obtention des vitesses vx et vy par la fonction polaire2xy
        v_r, v_theta = vitesse_polaire(psi_exact,nr,ntheta, prm)
         
        #réponse analytique du coefficient de pression
        cp_exact = 1 -4*np.sin(theta)**2
        
        #réponse numérique du coefficient de pression
        cp = coeff_pression(v_r[:,0], v_theta[:,0], prm)
        
        #vérification dimensions vecteur numérique
        assert(len(cp) == ntheta)
        
        #vérification des valeurs du vecteur numériques
        for i in range(0,ntheta):
            assert(all(abs(cp - cp_exact) < 1e-03))
            
   
    #test de la fonction coeff_aerodynamique et vérification des résultats numériques des coefficients de trainée et de portance
    def test_coeff_aerodynamique(self):
        #nombre de points utilisés
        nr = 411
        ntheta = 411
        
        #discrétisation des r et theta       
        r_mat,theta_mat = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
        theta = theta_mat[:,0]
        r = prm.R
        
        #initialisation des résultats analytiques de la fonction de courrant
        psi_exact = np.zeros(nr*ntheta)
        
        #boucles pour obtenir résultats analytiques de la fonction de courrant
        for i in range(nr):
            for j in range(ntheta):
                k = ij2k(i,j,ntheta)
                psi_exact[k] = prm.U_inf * r_mat[-1-j,i] * np.sin(theta_mat[-1-j,i]) * (1 - prm.R**2 / r_mat[-1-j,i]**2) 
                
        #obtention des vitesses vx et vy par la fonction polaire2xy                        
        v_r_mat, v_theta_mat = vitesse_polaire(psi_exact,nr,ntheta, prm)
        
        #réponse numérique des coefficients cd et cl
        cd, cl = coeff_aerodynamique(v_r_mat, v_theta_mat, nr, ntheta, prm)
        
        #valeur analytique des coefficients cd et cl
        cd_exact = 0
        cl_exact = 0
        
        #vérification des valeurs numériques de cd et cl
        assert(abs(cd - cd_exact) < 1e-05)
        assert(abs(cl - cl_exact) < 1e-05)
        
  
        
         
         
    