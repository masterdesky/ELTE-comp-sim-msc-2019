import random
import numpy as np
from numba import jit
from datetime import datetime


def initial_parameters():
    # Number of particles
    N = 264
    # Number of pairs
    n_pairs = int(N * (N - 1) / 2)
    # Tempreature
    T = 50
    # Side length of box
    L = None
    # Mean density of particles
    rho = 1.2
    # Time step length
    dt = 1e-3

    # Cut-off limit of Lennard-Jones Potential
    r_c = 2.5
    # Maximal size of pair-list
    r_max = 3.2
    # Update interval of updating the pair-list
    update_interval = 10

    # In the short description it is discussed, that
    # it is adviced to choose some physical quantities to 1
    # as it makes the calculations easier.
    m = 1
    V_0 = 1
    sigma = 1

    # Boundary conditions for the simulation
    # boundary = ['periodic' | 'bounded']
    boundary = 'bounded'

    # Natural constants
    # However calculating velocities, we set k_B to 1
    k_B = 1.38e-23    # Boltzmann constant [J/K]
    N_A = 6.022e23    # Avogadro's number [1/mol]
    
    return N, n_pairs, T, L, rho, dt, r_c, r_max, update_interval,\
           m, V_0, sigma, boundary, k_B, N_A

@jit(nopython=True)
def sign_choose():
    return -1 if np.random.random() > 0.5 else 1

@jit(nopython=True)
def instantaneous_temperature(v):
    vSqd = 0
    for i in range(N):
        for k in range(3):
            vSqd += v[i][k]**2
    return vSqd / (3 * (N - 1))

@jit(nopython=True)
def instantaneous_kinetic_E(v):
    vSqd = 0
    for i in range(N):
        for k in range(3):
            vSqd += v[i][k]**2
    
    return 1/2 * m * vSqd    

@jit(nopython=True)
def init_lattice_params(N, L, rho):
    
    if L is None and rho is None:
        raise AttributeError('Either \'rho\' or \'L\' should be given!')
    # Compute side of cube (L) from number of particles (N) and number density (rho)
    if L is None and rho is not None:
        L = (N / rho)**(1/3)
    # If L is given always use its value instead of rho
    elif L is not None and rho is None:
        rho = N / L**3
    else:
        rho = N / L**3

    # Find M large enough to fit N atoms on an fcc lattice
    M = 1
    while 4 * M**3 < N:
        M += 1

    # Lattice constant of conventional cell
    a_latt = L / M
    
    return L, rho, M, a_latt

@jit(nopython=True)
def compute_separation(r, i, j):
    # Find separation using closest image convention
    dr_sqd = 0
    dr = np.zeros((3))
    for k in range(3):
        dr[k] = r[i][k] - r[j][k]
        if boundary == 'periodic':
            # Closest image convention
            if dr[k] >= 0.5*L:
                dr[k] -= L
            if dr[k] < -0.5*L:
                dr[k] += L
        dr_sqd += dr[k]**2
    return dr, dr_sqd

@jit(nopython=True)
def update_pair_list(r):

    #print('Pre-update n_pairs :', n_pairs)
    pair_list = []

    # Start counting number of pairs from 0 again
    N_PAIRS = 0
    for i in range(N-1):
        for j in range(i+1, N):
            dr, dr_sqd = compute_separation(r, i, j)
            if dr_sqd < r_max**2:
                pair_list.append([i, j])
                N_PAIRS += 1
    pair_list = np.array(pair_list)
    #print('After-update n_pairs :', N_PAIRS)
    return pair_list, N_PAIRS

@jit(nopython=True)
def update_pair_separations(r, pair_list):

    dr_pair = np.zeros((len(pair_list),3))
    dr_sqd_pair = np.zeros(len(pair_list))

    for p in range(len(pair_list)):
        i = pair_list[p][0]
        j = pair_list[p][1]
        dr, dr_sqd = compute_separation(r, i, j)
        dr_pair[p] = dr
        dr_sqd_pair[p] = dr_sqd
        
    return dr_pair, dr_sqd_pair

@jit(nopython=True)
def init_positions():
    
    r = np.zeros((N,3))

    # 4 atomic positions in fcc unit cell 
    xCell = np.array((0.25, 0.75, 0.75, 0.25))
    yCell = np.array((0.25, 0.75, 0.25, 0.75))
    zCell = np.array((0.25, 0.25, 0.75, 0.75))

    # Atoms placed so far
    n = 0
    for x in range(M):
        for y in range(M):
            for z in range(M):
                for k in range(4):
                    if (n < N):
                        r[n][0] = (x + xCell[k]) * a_latt
                        r[n][1] = (y + yCell[k]) * a_latt
                        r[n][2] = (z + zCell[k]) * a_latt
                        n += 1
    return r

@jit(nopython=True)
def gasdev():
    available = False
    if available is False:
        while True:
            # Pick two uniform numbers in the square extending from -1 to +1 in each direction
            v1 = np.random.random() * sign_choose()
            v2 = np.random.random() * sign_choose()
            # See if they are inside the unit circle
            rsq = v1**2 + v2**2
            # If they are not, try again
            if rsq >= 1 or rsq == 0:
                continue
            else:
                break
        # Now make the Box-Müller transformation to get two normal deviates.
        # Return one and save the other for next time.
        fac = np.sqrt(-2 * np.log(rsq) / rsq)
        gset = v1 * fac
        available = True
        return v2 * fac
    else:
        available = False
        return gset

@jit(nopython=True)
def rescale_velocities(v):
    v_sqd_sum = 0
    for n in range(N):
        v_sqd_sum += np.sum(v[n]**2)
    lmbda = np.sqrt(3 * (N-1) * T / v_sqd_sum)
    v *= lmbda

    return v

@jit(nopython=True)
def init_velocities():
    
    v = np.zeros((N,3))
    
    # Gaussian with unit variance
    for n in range(N):
        for i in range(3):
            v[n][i] = gasdev()
    
    # Calculate velocity of CM
    # v_CM = sum_i v_i / N
    v_CM = np.zeros(3)
    for n in range(N):
        v_CM += v[n]
    v_CM /= N

    # Substract from velocities
    for n in range(N):
        v[n] -= v_CM

    # Rescale velocities to get the desired instantaneous temperature
    v = rescale_velocities(v)
    
    return v

@jit(nopython=True)
def compute_accelerations(pair_list, dr_pair, dr_sqd_pair):
    
    # Potential energy
    potential_E = 0
    
    # Virial theorem
    virial_E = 0

    # All accelerations at the start of the step are set to 0
    a = np.zeros((N,3))

    for p in range(n_pairs):
        i = int(pair_list[p][0])
        j = int(pair_list[p][1])
        if dr_sqd_pair[p] < r_c**2:
            # rx_inv = 1/r^x
            r2_inv = 1 / dr_sqd_pair[p]
            r6_inv = r2_inv**3
            r12_inv = r6_inv**2
            # Force between the particles with indeces 'i' and 'j'
            # F = 24 * V_0 * r2_inv * (2 * sigma^12 * r12_inv - sigma^6 * r6_inv) * |r|
            F_wo_r = 24 * V_0 * r2_inv * (2 * sigma**12 * r12_inv - sigma**6 * r6_inv)

            # Calculate accelerations of interacting particles
            # a = F/m
            # Since m = 1, we can write a = F
            for d in range(3):
                F = F_wo_r * dr_pair[p][d]
                a[i][d] += F
                a[j][d] -= F
                
                if i < j:
                    virial_E += r[i][d] * F

            # Step with potential energy
            # V = 4 * V_0 * (sigma^12 * r12_inv - sigma^6 * r6_inv)
            potential_E += 4 * V_0 * (sigma**12 * r12_inv - sigma**6 * r6_inv)
            
            

    return a, potential_E, virial_E

@jit(nopython=True)
def velocity_verlet(r, v, a, pair_list, dr_pair, dr_sqd_pair):

    # Verlet stepping rule for coordinates:
    # r_(n+1) = r_(n) + dt * v_(n) + 1/2 * dt^2 * a_(n)
    for i in range(N):
        for k in range(3):
            r[i][k] += dt * v[i][k] + 0.5 * dt**2 * a[i][k]

            if(boundary == 'periodic'):
                # Use periodic boundary conditions
                if (r[i][k] < 0):
                    r[i][k] += L
                if (r[i][k] >= L):
                    r[i][k] -= L

            # Verlet stepping rule for velocity
            # v_(n+1) = v_(n) + 1/2 * dt * a_(n+1) + {{1/2 * dt * a_(n)}}
            # First step with the {{a_(n)}} part
            v[i][k] += 0.5 * dt * a[i][k]

    # Calculate a_(n+1)
    dr_pair, dr_sqd_pair = update_pair_separations(r, pair_list)
    a, potential_E, virial_E = compute_accelerations(pair_list, dr_pair, dr_sqd_pair)

    for i in range(N):
        for k in range(3):
            if(boundary == 'bounded'):
                # use bounded boundary conditions
                if (r[i][k] < 0 or r[i][k] >= L):
                    v[i][k] *= -1
                    a[i][k] *= -1

            # Verlet stepping rule for velocity
            # v_(n+1) = v_(n) + {{1/2 * dt * a_(n+1)}} + 1/2 * dt * a_(n)
            # Now add the {{a_(n+1)}} part
            v[i][k] += 0.5 * dt * a[i][k]

    kinetic_E = instantaneous_kinetic_E(v)

    return r, v, a, pair_list, dr_pair, dr_sqd_pair, kinetic_E, potential_E, virial_E


# MAIN
# Number of simulated steps
n_steps = 2000

# Initial parameters
N, n_pairs, T, L, rho, dt, r_c, r_max, update_interval,\
                     m, V_0, sigma, boundary, k_B, N_A = initial_parameters()
L, rho, M, a_latt = init_lattice_params(N, L, rho)

# ---------- HISTORICAL PARAMETERS ----------
r_history = np.zeros((n_steps+1, N, 3))
v_history = np.zeros((n_steps+1, N, 3))
a_history = np.zeros((n_steps+1, N, 3))
kinetic_E_history = np.zeros(n_steps+1)
potential_E_history = np.zeros(n_steps+1)
virial_E_history = np.zeros(n_steps+1)
temperature_history = np.zeros(n_steps+1)

# ---------- INITIALIZATION ----------
# Starting values
r = init_positions()
v = init_velocities()
pair_list, n_pairs = update_pair_list(r)
dr_pair, dr_sqd_pair = update_pair_separations(r, pair_list)
kinetic_E = instantaneous_kinetic_E(v)
a, potential_E, virial_E = compute_accelerations(pair_list, dr_pair, dr_sqd_pair)

r_history[0] = r
v_history[0] = v
a_history[0] = a
kinetic_E_history[0] = kinetic_E
potential_E_history[0] = potential_E
virial_E_history[0] = virial_E
temperature_history[0] = instantaneous_temperature(v)


# ---------- STEPPING THE SIMULATION ----------
for i in range(n_steps):

    r, v, a, pair_list, dr_pair, dr_sqd_pair, kinetic_E, potential_E, virial_E =\
    velocity_verlet(r, v, a, pair_list, dr_pair, dr_sqd_pair)
    if i % 200 == 0:
        v = rescale_velocities(v)
        print('Current run: {0}/{1}'.format(i, n_steps))

    if i % update_interval == 0:
        #print('Pre-interval n_pairs :', n_pairs)
        #print('Pre-interval len(pair_list) :', len(pair_list))
        pair_list, n_pairs = update_pair_list(r)
        print('After-interval len(pair_list) :', len(pair_list))
        dr_pair, dr_sqd_pair = update_pair_separations(r, pair_list)


    r_history[i+1] = r
    v_history[i+1] = v
    a_history[i+1] = a
    kinetic_E_history[i+1] = kinetic_E
    potential_E_history[i+1] = potential_E
    virial_E_history[i+1] = virial_E
    temperature_history[i+1] = instantaneous_temperature(v)

print('Finished!')