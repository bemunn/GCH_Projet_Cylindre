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



#Initialisation de la liste de nombres de points à tester
N = np.arange(10,302,10)

#Initialisation des vecteurs d'erreur

erreur_moy_psi = np.zeros(len(nr))
erreur_moy_vr = np.zeros(len(nr))
erreur_moy_vtheta = np.zeros(len(nr))
erreur_moy_cp = np.zeros(len(nr))
erreur_moy_cd = np.zeros(len(nr))
erreur_moy_cl = np.zeros(len(nr))



#Boucle faisant varier le nombre de points de discrétisation
for a,n in enumerate(N):
    
    #Calcul de la valeur de psi
    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],n,n)
    psi_num = mdf(n,n, prm)
    
    #Initialisation du vecteur de psi exact
    psi_exact = np.zeros(n**2)

    #Boucle permettant de calculer psi en tout point i,j
    for i in range(n):
        for j in range(n):
            k = ij2k(i,j,n)
            psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)
    
    #Erreur sur psi pour un nombre de point n        
    erreur_moy_psi[a] = np.mean(abs(psi_exact - psi_num))
    
    #Calcul de la vitesse et des coefficients Cp, Cd et Cl à partir de psi numérique
    vr, vtheta = vitesse_polaire(psi_num,n,n, prm)
    cp = coeff_pression(vr[:,0], vtheta[:,0], prm)
    cd, cl = coeff_aerodynamique(vr, vtheta, n, n, prm)
    
    #Calcul des vitesses exactes
    vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
    vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)
    
    #Caclul des coefficients exacts
    cp_exact = -1 + 2 * (np.cos(theta[:,0])**2 - np.sin(theta[:,0])**2)
    cd_exact = 0
    cl_exact = 0
    
    
    #Calcul des erreurs sur les vitesses et les coefficients pour un nombre de points n
    erreur_moy_vr[a] = np.mean(abs(vr_exact - vr))
    erreur_moy_vtheta[a] = np.mean(abs(vtheta_exact - vtheta))
    erreur_moy_cp[a] = np.mean(abs(cp_exact - cp))
    erreur_moy_cd[a] = np.mean(abs(cd_exact - cd))
    erreur_moy_cl[a] = np.mean(abs(cl_exact - cl))


#Calcul des ordres de convergence des erreurs sur tous les paramètres
ordre_psi =abs(np.log10(erreur_moy_psi[-1])-np.log10(erreur_moy_psi[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2])) 
ordre_vr =abs(np.log10(erreur_moy_vr[-1])-np.log10(erreur_moy_vr[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2]))
ordre_vtheta =abs(np.log10(erreur_moy_vtheta[-1])-np.log10(erreur_moy_vtheta[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2])) 
ordre_cp =abs(np.log10(erreur_moy_cp[-1])-np.log10(erreur_moy_cp[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2]))
ordre_cd =abs(np.log10(erreur_moy_cd[-1])-np.log10(erreur_moy_cd[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2])) 
ordre_cl =abs(np.log10(erreur_moy_cl[-1])-np.log10(erreur_moy_cl[-2]))/abs(np.log10(nr[-1])-np.log10(nr[-2])) 


#Plot de l'erreur sur psi
plt.loglog(nr,erreur_moy_psi,'-g')
#Ajout d'une pente d'ordre 2 pour visualiser la convergence de l'ordre
plt.loglog(np.array([nr[0],nr[-1]]), np.array([erreur_moy_psi[-1]/(10 **(-abs(np.log10(nr[-1])-np.log10(nr[0])) * 2)) ,erreur_moy_psi[-1]]), '--r')
plt.legend([r'$|\psi_{th} - \psi_{num}|$', "Pente d'ordre 2"])
plt.title(r'Erreur moyenne de $\psi$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.ylabel('Erreur')
plt.xlim(nr[0],nr[-1])  # Fixer les limites en x
plt.grid(which='both')
plt.savefig('ErreurPsiNrNtheta.png', dpi= 1000)
plt.show()

#Plot de  l'erreur sur v_r et v_t
plt.loglog(nr,erreur_moy_vr,'-g')
#Ajout d'une pente d'ordre 2 pour visualiser la convergence de l'ordre
plt.loglog(np.array([nr[0],nr[-1]]), np.array([erreur_moy_vr[-1]/(10 **(-abs(np.log10(nr[-1])-np.log10(nr[0])) * 2)) ,erreur_moy_vr[-1]]), '--r')
plt.loglog(nr,erreur_moy_vtheta,'-b')
#Ajout d'une pente d'ordre 2 pour visualiser la convergence de l'ordre
plt.loglog(np.array([nr[0],nr[-1]]), np.array([erreur_moy_vtheta[-1]/(10 **(-abs(np.log10(nr[-1])-np.log10(nr[0])) * 2)) ,erreur_moy_vtheta[-1]]), '--', color = 'orange')
plt.title(r'Erreurs moyennes de $v_r$ et $v_{\theta}$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.legend([r'$|v_{r-th} - v_{r-num}|$', "Pente d'ordre 2", r'$|v_{\theta-th} - v_{\theta-num}|$', "Pente d'ordre 2"])
plt.ylabel('Erreur')
plt.xlim(nr[0],nr[-1])  # Fixer les limites en x
plt.grid(which='both')
plt.savefig('ErreurVitessesNrNtheta.png', dpi= 1000)
plt.show()

#Plot de l'erreur sur c_p
plt.loglog(nr,erreur_moy_cp,'-g')
#Ajout d'une pente d'ordre 2 pour visualiser la convergence de l'ordre
plt.loglog(np.array([nr[0],nr[-1]]), np.array([erreur_moy_cp[-1]/(10 **(-abs(np.log10(nr[-1])-np.log10(nr[0])) * 2)) ,erreur_moy_cp[-1]]), '--r')
plt.title(r'Erreurs moyennes de $C_p$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.legend([r'$|c_{p-th} - c_{p-num}|$', "Pente d'ordre 2"])
plt.ylabel('Erreur')
plt.xlim(nr[0],nr[-1])  # Fixer les limites en x
plt.grid(which='both')
plt.savefig('ErreurCoeffPressNrNtheta.png', dpi= 1000)
plt.show()

#Plot de l'erreur sur c_d et c_l
plt.loglog(nr,erreur_moy_cd,'-g')
plt.loglog(nr,erreur_moy_cl,'-r')
plt.title(r'Erreurs moyennes de $C_d$ et $C_l$ selon le nombre de points ($n_r$ = $n_{\theta}$)')
plt.xlabel(r'$N = n_r = n_{\theta}$')
plt.legend([r'$|c_{d-th} - c_{d-num}|$',r'$|c_{l-th} - c_{l-num}|$'])
plt.ylabel('Erreur')
plt.xlim(nr[0],nr[-1])  # Fixer les limites en x
plt.grid(which='both')
plt.savefig('ErreurCoeffAeroNrNtheta.png', dpi= 1000)
plt.show()

# Pas de pente d'ordre 2 à plot étant donnée la nature chaotique du comportement de l'erreur de ces deux coefficients