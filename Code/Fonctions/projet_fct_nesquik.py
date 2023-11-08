# Importation des modules
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def ij2k(i,j, ny):
    """ Fonction convertissant le système à deux indices i,j en système un indice unique k

    Entrées:
        - i : Indice qui n'est pas dirigé dans le même sens que k (x)
        - j : Indice qui est dirigé dans le même sens que k (y)
        - ntheta : Discrétisation de l'espace en theta (nombre de points)

    Sorties:
        - k : Indice unique indiquant la position dans le domaine étudié
    """

    k = j + i * ny

    return k

def k2ij(k, ny):
    """ Fonction convertissant le système à un indice unique k en un sytème à deux indices i,j

    Entrées:
        - k : Indice unique indiquant la position dans le domaine étudié
        - ntheta : Discrétisation de l'espace en theta (nombre de points)

    Sorties:
        - i : Indice qui n'est pas dirigé dans le même sens que k (x)
        - j : Indice qui est dirigé dans le même sens que k (y)
    """

    j = np.mod(k, ny)
    i = int(( k - j ) / ny)       # on veut que i et j soient des integers et non des floats pour indexer

    return i,j

def k2ij_matrix(ny, vect):
    """ Fonction convertissant un vecteur indexé en k en matrice indexée en i,j

    Entrées:
        - k : Indice unique indiquant la position dans le domaine étudié
        - ntheta : Discrétisation de l'espace en theta (nombre de points)
        - vect : vecteur indexé à l'aide de k

    Sorties:
        - mat : Matrice indexé en [i,j] tel que k = ny * i + j
    """
    k_max = len(vect) - 1             # correspond à l'index du dernier element du vecteur
    i_max, j_max = k2ij(k_max, ny) 

    mat = np.zeros([j_max+1,i_max+1]) # creation matrice aux dimensions adequates

    for k in range(len(vect)):        # on place tous les elements du vecteur a l'endroit associé dans la matrice
        i, j  = k2ij(k, ny)
        mat[j, i] = vect[k] # j vers le haut ou bas?

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


def resolution_temps_explicite(nx, ny, M, L, T, D, xi, xf, yi, yf, dt,tf):
    
    S = M/(L*T)
    x_vec = np.linspace(xi,xf,nx)
    y_vec = np.linspace(yi,yf,ny)
    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    
    Ct = np.zeros([ny,nx])
    Ct[0, :] = S
    t = 0
    
    
    Ctdt = np.zeros([ny,nx])
    while t < tf:
        for i in range(1, len(x_vec)-1):
            for j in range(1,len(y_vec)-1):
                if t > T:
                    S = 0
                    
                Ctdt[j,i] =   (1 - 2*dt*D*(1/dx**2 + 1/dy**2)) * Ct[j,i] + (dt*D/dx**2)*(Ct[j, i+1] + Ct[j, i-1]) + (dt*D/dy**2)*(Ct[j-1, i] + Ct[j+1, i]) + S    
                
        Ct = Ctdt
        t = t+dt
        
    
    return 

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



    


    