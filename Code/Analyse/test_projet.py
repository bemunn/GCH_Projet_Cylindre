# Importation des modules
# import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pytest



sys.path.append('../Fonctions')

try:
    from projet_fct import *
except:
    pass


#------------------------------------------------------------------------------
# Code principal pour l'analyse des résultats
# Il faudra faire appel aux fonctions programmées dans projet_fct.py afin de
# calculer différents éléments. Des graphiques seront être générés pour
# visualiser les résultats.
#------------------------------------------------------------------------------

# Assignation des paramètres

class parametres():
    U_inf = 1       # Vitesse du fluide éloigné du cylindre [-]
    R     = 1       # Rayon interne du cylindre creux [-]
    R_ext = 5       # Rayon externe du cylindre creux [-]   

prm = parametres()

nr = 501
ntheta = 501

psi_num = mdf_assemblage(nr,ntheta, prm)

psi_exact = np.zeros(nr*ntheta)

r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

for i in range(nr):
    for j in range(ntheta):
        k = ij2k(i,j,ntheta)
        psi_exact[k] = prm.U_inf * r[-1-j,i] * np.sin(theta[-1-j,i]) * (1 - prm.R**2 / r[-1-j,i]**2)

erreur_moy = np.mean(abs(psi_exact - psi_num))

print(erreur_moy)

vr, vtheta = vitesse_polaire(psi_num,nr,ntheta, prm)
 
vr_exact = prm.U_inf*np.cos(theta)*(1-prm.R**2/r**2)
vtheta_exact = -np.sin(theta)*prm.U_inf*(1+prm.R**2/r**2)

erreur_vr = np.mean(abs(vr_exact - vr))
erreur_vtheta = np.mean(abs(vtheta_exact - vtheta))

cd, cl = coeff_aerodynamique(vr, vtheta, nr, ntheta, prm)

V = np.sqrt(np.square(vr) + np.square(vtheta))

vx, vy = polaire2xy(vr, vtheta, nr,ntheta, prm)
x = r * np.cos(theta)
y = r* np.sin(theta)


# fig, ax1 = plt.subplots(constrained_layout=True)

# fig11 = ax1.pcolormesh(x,y, vx)


# fig2, ax2 = plt.subplots(constrained_layout=True)
# fig12 = ax2.pcolormesh(x,y, vy)


circle1 = plt.Circle((0, 0), 1, color='white', fill=1)
circle2 = plt.Circle((0, 0), 5, color='lightgrey', fill=1)
fig4, ax4 = plt.subplots(constrained_layout=True)
ax4.add_patch(circle2)
ax4.add_patch(circle1)
# ax4.grid(True, which="both")
bonds_r = 16
bonds_theta = 2
ax4.quiver(x[::bonds_theta,::bonds_r ], y[::bonds_theta , ::bonds_r], vx[::bonds_theta , ::bonds_r], vy[::bonds_theta , ::bonds_r], units = "dots", color = "red", scale = .006, label = "Vecteurs vitesse") 
ax4.set_title("Champ de vitesse d'un fluide en écoulement autour\nd'un cylindre en régime permanent", size=12)
ax4.set_xlabel(r'Position en $x$ [-]', size=13)
ax4.set_ylabel(r'Position en $y$ [-]', size=13)
ax4.axis([-5.1, 5.1, -5.1, 5.1])
fig4.savefig("champ_de_vitesse.png", dpi=900)
#%%
psi_num_mat = np.abs(k2ij_matrix(ntheta, psi_num))
fig22, ax22 = plt.subplots(constrained_layout=False)
contour = ax22.contourf(x, y, psi_num_mat, 15, cmap='plasma')
cbar =plt.colorbar(contour, ax=ax22)
cbar.set_label(r'$\psi$ ', rotation=0, labelpad=10, size=12)
for key, spine in ax22.spines.items():
    spine.set_visible(False)
ax22.set_xlabel(r'Position en $x$ [-]', size=13)
ax22.set_ylabel(r'Position en $y$ [-]', size=13)
# ax22.grid(True, which="both")

# fig22.savefig("ligne_courant.png", dpi=1200)

plt.show()

# %%
pytest.main(['-q', '--tb=long', 'Correction.py'])