# Importation des modules
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def ij2k(i,j, ntheta):
    """ Fonction convertissant le système à deux indices i,j en système un indice unique k

    Entrées:
        - i : Indice qui n'est pas dirigé dans le même sens que k (x)
        - j : Indice qui est dirigé dans le même sens que k (y)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)

    Sorties:
        - k : Indice unique indiquant la position dans le domaine étudié
    """

    k = j + i * ntheta

    return k

def k2ij(k, ntheta):
    """ Fonction convertissant le système à deux indices i,j en système un indice unique k

    Entrées:
        - k : Indice unique indiquant la position dans le domaine étudié
        - ntheta : Discrétisation de l'espace en theta (nombre de points)

    Sorties:
        - i : Indice qui n'est pas dirigé dans le même sens que k (x)
        - j : Indice qui est dirigé dans le même sens que k (y)
    """

    j = np.mod(k, ntheta)
    i = int(( k - j ) / ntheta)

    return i,j

def position(X,Y,nx,ny):
    """ Fonction générant deux matrices de discrétisation de l'espace

    Entrées:
        - X : Bornes du domaine en x, X = [x_min, x_max]
        - Y : Bornes du domaine en y, Y = [y_min, y_max]
        - nx : Discrétisation de l'espace en x (nombre de points)
        - ny : Discrétisation de l'espace en y (nombre de points)

    Sorties (dans l'ordre énuméré ci-bas):
        - x : Matrice (array) de dimension (ny x nx) qui contient la position en x
        - y : Matrice (array) de dimension (ny x nx) qui contient la position en y
            * Exemple d'une matrice position :
            * Si X = [-1, 1] et Y = [0, 1]
            * Avec nx = 3 et ny = 3
                x = [-1    0    1]
                    [-1    0    1]
                    [-1    0    1]

                y = [1    1    1  ]
                    [0.5  0.5  0.5]
                    [0    0    0  ]
    """

    x_ligne = np.linspace(X[0], X[1], nx)
    x = np.tile(x_ligne, [ny,1])
    
    y_ligne = np.linspace(Y[1], Y[0], ny)
    y_colonne = y_ligne.reshape(len(y_ligne), 1)
    y = np.tile(y_colonne, [1,nx])

    return x, y


def mdf_assemblage(nr,ntheta, prm):
    """ Fonction assemblant la matrice A et le vecteur b

    Entrées:
        - nr : Discrétisation de l'espace en r (nombre de points)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sorties (dans l'ordre énuméré ci-bas):
        - A : Matrice (array)
        - b : Vecteur (array)
    """

    # Fonction à écrire

    N = nr * ntheta 
    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

    dr = abs(r[-1,-1] - r[-1,-2])
    dtheta = abs(theta[-1,-1] - theta[-2,-1])

    A = np.zeros([N,N]) 
    b = np.zeros(N)

    ## Fonction 
    for i in range(1,nr-1):
        for j in range(1,ntheta-1):
            k = ij2k(i,j, ntheta)

            b[k]       = 0
            A[k,k]     = -2 * (dtheta**2 + dr**2 / r[-1-j,i]**2)
            A[k, k-1]  = dr**2 / r[-1-j,i]**2
            A[k, k+1]  = dr**2 / r[-1-j,i]**2
            A[k, k-ntheta] = dtheta**2 * (1 - dr / (2 * r[-1-j,i]))
            A[k, k+ntheta] = dtheta**2 * (1 + dr / (2 * r[-1-j,i]))
   
    
    ## Conditions frontières
    # Condition haut/bas
    
    for i in range(nr):
        j= ntheta - 1
        k = ij2k(i,j, ntheta)
        A[k,k] = 1
        b[k]   = 0

        j= 0
        k = ij2k(i,j, ntheta)
        A[k,k] = 1
        b[k]   = 0        

    
    #  Conditions gauche
    for j in range(ntheta):
        i = 0
        k = ij2k(i,j, ntheta)
        A[k,k] = 1
        b[k]   = 0

    #  Conditions droite
    for j in range(ntheta):
        i = nr - 1
        k = ij2k(i,j, ntheta)
        A[k,k] = 1
        b[k]   = prm.U_inf * prm.R_ext * (1 - prm.R**2 / prm.R_ext**2) * np.sin(theta[-1-j,i])

    psi = np.linalg.solve(A,b)

    return psi

def vitesse_polaire(psi,nr,ntheta, prm ):
    """ Fonction assemblant la matrice A et le vecteur b

    Entrées:
        - psi : Vecteur (array) de taille N contenant les valeurs de phi à chaque point
        - nr : Discrétisation de l'espace en r (nombre de points)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sorties (dans l'ordre énuméré ci-bas):
        - v_r : Matrice (array) nr x ntheta contenant la vitesse selon l'axe r à chaque point [-]
        - v_theta : Matrice (array) nr x ntheta contenant la vitesse selon l'axe theta à chaque point [-]
    """

    v_r = np.zeros([ntheta, nr])
    v_theta = np.zeros([ntheta, nr])

    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

    dr = abs(r[-1,-1] - r[-1,-2])
    dtheta = abs(theta[-1,-1] - theta[-2,-1])

    #v_r
    
    for i in range(nr):
        j = 0
        k = ij2k(i,j, ntheta)
        v_r[-j-1, i] = (1 / r[-j-1, i]) * (-psi[k+2] + 4 * psi[k+1] - 3 * psi[k]) / (2 * dr)

        j = ntheta - 1
        k = ij2k(i,j, ntheta)
        v_r[-j-1, i] = (1 / r[-j-1, i]) * (psi[k-2] - 4 * psi[k-1] + 3 * psi[k]) / (2 * dr)

        for j in range(1, ntheta - 1):
            k = ij2k(i,j, ntheta)
            v_r[-j-1, i] = (1 / r[-j-1, i]) * (psi[k+1] - psi[k-1]) / (2 * dr)


    #v_theta
    
    for j in range(ntheta):
        i=0
        k = ij2k(i,j, ntheta)
        v_theta[-j-1, i] = - (-psi[k+2*ntheta] + 4 * psi[k+ntheta] - 3 * psi[k]) / (2 * dr)

        i = nr - 1
        k = ij2k(i,j, ntheta)
        v_theta[-j-1, i] = - (psi[k-2*ntheta] - 4 * psi[k-ntheta] + 3 * psi[k]) / (2 * dr)

        for i in range(1, nr - 1):
            k = ij2k(i,j, ntheta)
            v_theta[-j-1, i] = - (psi[k+ntheta] - psi[k-ntheta]) / (2 * dr)

    return v_r, v_theta


def polaire2xy(v_r, v_theta, nr,ntheta, prm):
    """ Fonction assemblant la matrice A et le vecteur b

    Entrées:
        - v_r : Matrice (array) nr x ntheta contenant la vitesse selon l'axe r à chaque point [-]
        - v_theta : Matrice (array) nr x ntheta contenant la vitesse selon l'axe theta à chaque point [-]
        - nr : Discrétisation de l'espace en r (nombre de points)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sorties (dans l'ordre énuméré ci-bas):
        - v_x : Matrice (array) nr x ntheta contenant la vitesse selon l'axe x à chaque point [-]
        - v_y : Matrice (array) nr x ntheta contenant la vitesse selon l'axe y à chaque point [-]
    """

    r,theta = position([prm.R, prm.R_ext],[0. , 2 * 1 * np.pi],nr,ntheta)

    v_x = np.zeros([ntheta, nr])
    v_y = np.zeros([ntheta, nr])

    for i in range(nr):
        for j in range(ntheta):
                v_x[-j-1, i], v_y[-1-j, i] = np.matmul([[np.cos(theta[-1-j,i]),-np.sin(theta[-1-j,i])],[np.sin(theta[-1-j,i]),np.cos(theta[-1-j,i])]], [v_r[-j-1, i],v_theta[-j-1,i] ])

    return v_x, v_y