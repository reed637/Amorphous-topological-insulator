import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Generate random points in a 2D space
N = 100 # Number of points
radius = 5.0 # Size of the box
r_cut = 1.5 # Cutoff radius

# Generate random positions
points = []
while len(points) < N:
    x,y = np.random.uniform(-radius, radius, 2)
    if np.sqrt(x**2 + y**2) <= radius:
        points.append((x, y))
positions = np.array(points)
# Create a KDTree for efficient neighbor search
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
plt.savefig('random_lattice_edge.png', dpi=300)
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
plt.title(f'Edge Eigenstate {idx} Probability Density')
plt.axis('equal')
plt.savefig(f'edge eigenstate_{idx}.png', dpi=300)
# Save the positions and neighbors to a file

r = np.linalg.norm(positions, axis=1)
r_max = np.max(r)
edge_threshold = 0.85 * r_max

print("Eigenstate | Energy | Edge Weight")
for i in range(k):
    psi = np.abs(eigenvectors[:, i])**2
    edge_weight = np.sum(psi[r > edge_threshold])
    print(f"{i} | {eigenvalues[i]:.4f} | {edge_weight:.3f}")

edge_data = [(idx, eigenvalues[idx], np.sum(np.abs(eigenvectors[:, idx])**2*[r > edge_threshold])) for idx in range(len(eigenvalues))]

edge_data.sort(key=lambda x: x[2], reverse=True)

for rank, (idx, energy, edge_weight) in enumerate(edge_data[:2]):
    psi = np.abs(eigenvectors[:, idx])**2
    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1], c=psi, s=30, cmap='plasma')
    plt.colorbar(label='Probability Density')
    plt.title(f'Top edge state {rank+1} (Energy: {energy:.4f}, Edge Weight: {edge_weight:.3f})')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'top_edge_state_{rank+1}.png', dpi=300)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]

    edge_weights_sorted = np.array([np.sum(np.abs(eigenvectors[:, i])**2*[r > edge_threshold]) for i in sorted_indices])

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(range(len(eigenvalues)), sorted_eigenvalues, c=edge_weights_sorted, cmap='plasma', s=25)
    plt.colorbar(sc, label='Edge Weight')
    plt.xlabel('Eigenstate Index')
    plt.ylabel('Energy')
    plt.title('Energy Spectrum with Edge Localization')
    plt.tight_layout()
    plt.savefig('energy_spectrum_edge_localization.png', dpi=300)

    from bott_index import compute_bott_index
    bott = compute_bott_index(H_csr.toarray(), positions, fermi_level=0.1)
    print(f"Bott index: {bott:.4f}")