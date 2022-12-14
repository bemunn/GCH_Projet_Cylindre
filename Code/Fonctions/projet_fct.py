# Importation des modules
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
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
    """ Fonction convertissant le système à un indice unique k en un sytème à deux indices i,j

    Entrées:
        - k : Indice unique indiquant la position dans le domaine étudié
        - ntheta : Discrétisation de l'espace en theta (nombre de points)

    Sorties:
        - i : Indice qui n'est pas dirigé dans le même sens que k (x)
        - j : Indice qui est dirigé dans le même sens que k (y)
    """

    j = np.mod(k, ntheta)
    i = int(( k - j ) / ntheta)       # on veut que i et j soient des integers et non des floats pour indexer

    return i,j

def k2ij_matrix(ntheta, vect):
    """ Fonction convertissant un vecteur indexé en k en matrice indexée en i,j

    Entrées:
        - k : Indice unique indiquant la position dans le domaine étudié
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - vect : vecteur indexé à l'aide de k

    Sorties:
        - mat : Matrice indexé en [i,j] tel que k = ny * i + j
    """
    k_max = len(vect) - 1             # correspond à l'index du dernier element du vecteur
    i_max, j_max = k2ij(k_max, ntheta) 

    mat = np.zeros([j_max+1,i_max+1]) # creation matrice aux dimensions adequates

    for k in range(len(vect)):        # on place tous les elements du vecteur a l'endroit associé dans la matrice
        i, j  = k2ij(k, ntheta)
        mat[-j-1, i] = vect[k] 

    return mat

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

    x_ligne = np.linspace(X[0], X[1], nx)        # vecteur positions x
    x = np.tile(x_ligne, [ny,1])                 # on copie le vecteur y fois
    
    y_ligne = np.linspace(Y[1], Y[0], ny)        # vecteur positions y
    y_colonne = y_ligne.reshape(len(y_ligne), 1) # vecteur vertical
    y = np.tile(y_colonne, [1,nx])               # on copie le vecteur x fois

    return x, y


def mdf(nr,ntheta, prm):
    """ Fonction assemblant la matrice A et le vecteur b et résolvant 
        pour psi à l'aide de matrices creuses

    Entrées:
        - nr : Discrétisation de l'espace en r (nombre de points)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sorties (dans l'ordre énuméré ci-bas):
        - psi : Vecteur (array)
    """

    # Fonction à écrire

    N = nr * ntheta 
    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

    dr = abs(r[-1,-1] - r[-1,-2])
    dtheta = abs(theta[-1,-1] - theta[-2,-1])

    A = lil_matrix((N,N))               # matrice creuse
    # A = np.zeros([N,N])
    b = np.zeros(N)

    ## Fonction discrétisée à l'intérieur du domaine étudié
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
    
    A = lil_matrix.tocsr(A)
    psi = spsolve(A,b)                  # Résolution du système linéaire

    return psi

def vitesse_polaire(psi,nr,ntheta, prm ):
    """ Fonction calculant les vitesses en tout point en coordonnées polaires

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

    # Fonctions discrétisées pour vr
    
    for i in range(nr):
        # Condition limite en bas
        j = 0
        k = ij2k(i,j, ntheta)
        v_r[-j-1, i] = (1 / r[-j-1, i]) * (-psi[k+2] + 4 * psi[k+1] - 3 * psi[k]) / (2 * dtheta)

        # Condition limite en haut
        j = ntheta - 1
        k = ij2k(i,j, ntheta)
        v_r[-j-1, i] = (1 / r[-j-1, i]) * (psi[k-2] - 4 * psi[k-1] + 3 * psi[k]) / (2 * dtheta)

        # Condition milieu
        for j in range(1, ntheta - 1):
            k = ij2k(i,j, ntheta)
            v_r[-j-1, i] = (1 / r[-j-1, i]) * (psi[k+1] - psi[k-1]) / (2 * dtheta)


    # Fonctions discrétisées pour vtheta
    
    for j in range(ntheta):
        # Condition limite à gauche
        i=0
        k = ij2k(i,j, ntheta)
        v_theta[-j-1, i] = - (-psi[k+2*ntheta] + 4 * psi[k+ntheta] - 3 * psi[k]) / (2 * dr)

        # Condition limite à droite
        i = nr - 1
        k = ij2k(i,j, ntheta)
        v_theta[-j-1, i] = - (psi[k-2*ntheta] - 4 * psi[k-ntheta] + 3 * psi[k]) / (2 * dr)

        # Condition milieu
        for i in range(1, nr - 1):
            k = ij2k(i,j, ntheta)
            v_theta[-j-1, i] = - (psi[k+ntheta] - psi[k-ntheta]) / (2 * dr)

    return v_r, v_theta


def polaire2xy(v_r, v_theta, nr,ntheta, prm):
    """ Fonction convertissant les vitesses en coordonnées polaires en vitesses en coordonnées cartésiennes
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

    r,theta = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)

    v_x = np.zeros([ntheta, nr])
    v_y = np.zeros([ntheta, nr])

    # Utilisation de la matrice de rotation
    for i in range(nr):
        for j in range(ntheta):
                v_x[-j-1, i], v_y[-1-j, i] = np.matmul([[np.cos(theta[-1-j,i]),-np.sin(theta[-1-j,i])],[np.sin(theta[-1-j,i]),np.cos(theta[-1-j,i])]], [v_r[-j-1, i],v_theta[-j-1,i] ])

    return v_x, v_y

def coeff_pression(vr_vect, vtheta_vect, prm):
    """ Fonction calculant le coefficient de pression Cp à partir des vecteurs vitesses

    Entrées:
        - vr_vect : Vecteur (array) de taille ntheta contenant la vitesse selon l'axe r à chaque theta à r = R [-]
        - vtheta_vect : Vecteur (array) de taille ntheta contenant la vitesse selon l'axe theta à chaque theta à r = R [-]
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sorties (dans l'ordre énuméré ci-bas):
        - cp : Vecteur (array) de taille ntheta contenant c_p à chaque theta à r = R [-]
    """
    
    V = np.sqrt(vr_vect**2 + vtheta_vect**2) # Norme de la vitesse

    cp = 1 - (V/prm.U_inf)**2
    
    return cp

def trapeze(x,y):
    """Fonction qui calcule l'intégrale avec la méthode des trapèzes

    Entrées:
        - x : Valeurs de l'abscisse, vecteur (array)
        - y : Valeurs de l'ordonnée, vecteur (array)

    Sortie:
        - S : Valeur de l'intégrale calculée (float)
    """

    S = 0       # Initialisation somme
    for i in range(int(len(x)) - 1):
        S = S + 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i]) 

    return S

def coeff_aerodynamique(v_r, v_theta, nr, ntheta, prm):
    """Fonction qui calcule le coefficient de trainée et de portance avec les vitesses

    Entrées:
        - v_r : Matrice (array) nr x ntheta contenant la vitesse selon l'axe r à chaque point [-]
        - v_theta : Matrice (array) nr x ntheta contenant la vitesse selon l'axe theta à chaque point [-]
        - nr : Discrétisation de l'espace en r (nombre de points)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - prm : Objet class parametres()
            - U_inf : Vitesse du fluide éloigné du cylindre [-]
            - R : Rayon interne du cylindre creux [-]
            - R_ext : Rayon externe du cylindre creux [-]

    Sortie:
        - cd : Valeur (float) du coefficient de trainée [-]
        - cl : Valeur (float) du coefficient de portance [-]
    """ 
    r,theta_mat = position([prm.R, prm.R_ext],[0. , 2 * np.pi],nr,ntheta)
    
    theta = np.flip(theta_mat[:,0])
    
    cp = np.flip(coeff_pression(v_r[:,0], v_theta[:,0], prm))
    
    cd = -0.5 * trapeze(theta, cp * np.cos(theta))
    
    cl = -0.5 * trapeze(theta, cp * np.sin(theta))
    
    return cd, cl
    


    