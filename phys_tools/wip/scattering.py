import numpy as np
# import scipy as sp
from numba import jit


@jit
def findforces(current_positions, masses, origins, origin_mass=2., energy=1.):
    S = masses
    pos = current_positions
    forces_x = np.zeros_like(S)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[1]):
        delta[:, :, i] = pos[:, i, None] - pos[:, i]
    distance = np.sqrt((delta ** 2.).sum(axis=-1))

    # compute the force magnitudes in the x direction for each pair point.
    # this could be parallelized

    for i in range(len(distance)):
        for j in range(len(distance)):

            if i != j:  # obviously we're infinitely close to itself.
                s1 = S[i]
                s2 = S[j]
                d_x, d_y = delta[i, j, :]
                θ = np.arctan(d_x / d_y)
                r = distance[i, j]
                r = max(r, .001)
                F = (s1 * s2) / r ** 2
                sign = d_x / np.abs(d_x)
                F_x = F * np.cos(θ) * sign

                forces_x[i] += F_x * energy
                #     forces_x *= energy
    for i in range(len(distance)):
        displacement = current_positions[i, 0] - origins[i]

        r = displacement
        s1 = S[i]
        F2 = s1 * origin_mass * (r * 10) ** 2

        sign = displacement / np.abs(displacement)
        forces_x[i] -= F2 * sign
    return forces_x


@jit
def drag(V, ρ=20000., Cd=.5):
    if V:
        sign = V / np.abs(V)
        #         Cd = .5 # approx for sphere
        #         ρ = 50.
        F_drag = sign * 0.5 * ρ * Cd * V ** 2

    else:
        F_drag = 0.
    # return 0.
    return F_drag


@jit
def accelerate(current_velocities, masses, forces, timestep):
    velos = np.zeros_like(current_velocities)
    for i in range(len(current_velocities)):
        v = current_velocities[i]
        F_g = forces[i]
        F_drag = drag(v)
        m = masses[i]
        F_total = F_g - F_drag
        a = F_total / 10.
#         if np.isnan(a):
#             print ('V: {:4e} F: {:4e} Drag: {:4e}'.format(v, F_g, F_drag))
        v2 = v + a * timestep
        velos[i] = v2
    return velos


@jit
def move(start_positions, velocities, timestep):
    new_positions = start_positions.copy()
    for i in range(len(velocities)):
        v = velocities[i]
        new_positions[i, 0] += v * timestep
    return new_positions


# @jit
def iterate(current_velocities, current_positions, masses, origins, origin_mass=200., timestep=.001, energy=1.):
    forces = findforces(current_positions, masses, origins, origin_mass=origin_mass, energy=energy)

    velos = accelerate(current_velocities, masses, forces, timestep)
    positions = move(current_positions, velos, timestep)
    return positions, velos