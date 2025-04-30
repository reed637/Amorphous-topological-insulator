import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Generate random points in a 2D space
N = 100 # Number of points
L = 10 # Size of the box
r_cut = 1.5 # Cutoff radius

# Generate random points
np.random.seed(0)
positions = np.random.rand(N, 2) * L
# Create a KDTree for efficient neighbor searching
tree = KDTree(positions)
neighbors = tree.query_pairs(r_cut)

plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], s=20, color='black')

for i, j in neighbors:
    x  = [positions[i, 0], positions[j, 0]]
    y  = [positions[i, 1], positions[j, 1]]
    plt.plot(x, y, color='gray', lw=0.5)
plt.title('Random Lattice with Neighbors links')
plt.axis('equal')
plt.savefig('random_lattice.png', dpi=300)
np.savetxt('positions.txt', positions)
np.savetxt('neighbors.txt', np.array(list(neighbors)), fmt='%d')
np.savetxt('positions.txt', positions)

from scipy.sparse import lil_matrix

#initialize a sparse Hamiltonian
H = lil_matrix((N, N), dtype=complex)

t = 1.0 # Hopping parameter
phi = np.pi / 2 # Phase factor #phase angle (adds topology)

# complex hopping to the neighbors
for i, j in neighbors:
    dx, dy = positions[j] - positions[i]
    angle = np.arctan2(dy, dx) # angle from i to j
    phase = np.exp(1j * phi * angle) # complex phase
    H[i, j] = -t * phase # Hopping from j to i
    H[j, i] = -t * np.conjugate(phase) # Hermitian conjugate 

from scipy.sparse.linalg import eigsh

# convert to CSR format for efficient computation
H_csr = H.tocsr()

# Compute the eigenvalues and eigenvectors
k = 10  # Number of eigenvalues to compute(energy levels)
eigenvalues, eigenvectors = eigsh(H_csr, k=k, which='SM')

print("lowest energy levels:")
print(np.round(eigenvalues, 4))

# Plot the eigenvalues
idx = 5
psi = np.abs(eigenvectors[:, idx])**2
plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], c=psi, s=30, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.title(f'Eigenstate {idx} Probability Density')
plt.axis('equal')
plt.savefig(f'eigenstate_{idx}.png', dpi=300)
# Save the positions and neighbors to a file
