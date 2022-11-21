# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:54:55 2022

@author: bemun
"""

# Importation des modules
# import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


sys.path.append('../Fonctions')

try:
    from projet_fct import *
except:
    pass

class parametres():
    U_inf = 1       # Vitesse du fluide éloigné du cylindre [-]
    R     = 1       # Rayon interne du cylindre creux [-]
    R_ext = 5       # Rayon externe du cylindre creux [-]   
    

prm = parametres()




nr = np.arange(11,112,10)
erreur_moy = np.zeros(len(nr))



for a,n in enumerate(nr):
    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],n,n)
    psi_num = mdf_assemblage(n,n, prm)
    psi_exact = np.zeros(n**2)
    print(n)
    for i in range(n):
        for j in range(n):
            k = ij2k(i,j,n)
            psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
            
    erreur_moy[a] = np.mean(abs(psi_exact - psi_num))
    
    
ordre =abs(np.log10(erreur_moy[-1])-np.log10(erreur_moy[0]))/abs(np.log10(nr[-1])-np.log10(nr[0]))  
plt.loglog(nr,erreur_moy,'-r')
plt.title(r'Erreur sur la valeur de $\psi$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel('N')
plt.ylabel('Erreur')
plt.grid(which='both')
plt.savefig('ErreurNrNtheta.png', dpi= 1000)
plt.show()