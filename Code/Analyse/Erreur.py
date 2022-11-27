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




nr = np.arange(11,302,10)
erreur_moy_psi = np.zeros(len(nr))
erreur_moy_vr = np.zeros(len(nr))
erreur_moy_vtheta = np.zeros(len(nr))
erreur_moy_cp = np.zeros(len(nr))
erreur_moy_cd = np.zeros(len(nr))
erreur_moy_cl = np.zeros(len(nr))



for a,n in enumerate(nr):
    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],n,n)
    psi_num = mdf_assemblage(n,n, prm)
    psi_exact = np.zeros(n**2)

    print(n)
    for i in range(n):
        for j in range(n):
            k = ij2k(i,j,n)
            psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
            
    erreur_moy_psi[a] = np.mean(abs(psi_exact - psi_num))
    vr, vtheta = vitesse_polaire(psi_num,n,n, prm)
    cp = coeff_pression(vr[:,0], vtheta[:,0], prm)
    cd, cl = coeff_aerodynamique(vr, vtheta, n, n, prm)
    vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
    vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)
    cp_exact = -1 + 2 * (np.cos(theta[:,0])**2 - np.sin(theta[:,0])**2)
    cd_exact = 0
    cl_exact = 0
    erreur_moy_vr[a] = np.mean(abs(vr_exact - vr))
    erreur_moy_vtheta[a] = np.mean(abs(vtheta_exact - vtheta))
    erreur_moy_cp[a] = np.mean(abs(cp_exact - cp))
    erreur_moy_cd[a] = np.mean(abs(cd_exact - cd))
    erreur_moy_cl[a] = np.mean(abs(cl_exact - cl))

ordre_psi =abs(np.log10(erreur_moy_psi[-1])-np.log10(erreur_moy_psi[0]))/abs(np.log10(nr[-1])-np.log10(nr[0])) 
ordre_vr =abs(np.log10(erreur_moy_vr[-1])-np.log10(erreur_moy_vr[0]))/abs(np.log10(nr[-1])-np.log10(nr[0]))
ordre_vtheta =abs(np.log10(erreur_moy_vtheta[-1])-np.log10(erreur_moy_vtheta[0]))/abs(np.log10(nr[-1])-np.log10(nr[0])) 
ordre_cp =abs(np.log10(erreur_moy_cp[-1])-np.log10(erreur_moy_cp[0]))/abs(np.log10(nr[-1])-np.log10(nr[0])) 
ordre_cd =abs(np.log10(erreur_moy_cd[-1])-np.log10(erreur_moy_cd[0]))/abs(np.log10(nr[-1])-np.log10(nr[0])) 
ordre_cl =abs(np.log10(erreur_moy_cl[-1])-np.log10(erreur_moy_cl[0]))/abs(np.log10(nr[-1])-np.log10(nr[0])) 


#plot erreur psi
plt.loglog(nr,erreur_moy_psi,'-r')
plt.title(r'Erreur sur la valeur de $\psi$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.ylabel(r'$|\psi_{th} - \psi_{num}|$')
plt.grid(which='both')
plt.savefig('ErreurPsiNrNtheta.png', dpi= 1000)
plt.show()

#plot erreur v_r et v_t
plt.loglog(nr,erreur_moy_vr,'-b')
plt.loglog(nr,erreur_moy_vtheta,'-g')
plt.title(r'Erreur sur la valeur de $v_r$ et $v_{\theta}$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.legend([r'$|v_{r-th} - v_{r-num}|$', r'$|v_{\theta-th} - v_{\theta-num}|$'])
plt.ylabel('Erreur')
plt.grid(which='both')
plt.savefig('ErreurVitessesNrNtheta.png', dpi= 1000)
plt.show()

#plot erreur c_p, c_d et c_l
plt.loglog(nr,erreur_moy_cp,'-b')
plt.loglog(nr,erreur_moy_cd,'-g')
plt.loglog(nr,erreur_moy_cd,'-r')
plt.title(r'Erreur sur la valeur de $c_p$, $c_d$ et $c_l$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.legend([r'$|c_{p-th} - c_{p-num}|$', r'$|c_{d-th} - c_{d-num}|$',r'$|c_{l-th} - c_{l-num}|$'])
plt.ylabel('Erreur')
plt.grid(which='both')
plt.savefig('ErreurCoeffAeroNrNtheta.png', dpi= 1000)
plt.show()