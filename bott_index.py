import numpy as np
from scipy.linalg import expm, logm

def compute_bott_index(H, positions, fermi_level=None):
    assert H.ndim == 2 and H.shape[0] == H.shape[1], "H must be a square matrix"
    eigvals, eigvecs = np.linalg.eigh(H)

    if fermi_level is None:
        fermi_level = (eigvals.min() + eigvals.max()) / 2
    occupied = eigvals < fermi_level

    if np.sum(occupied) == 0:
        raise ValueError("No occupied states found below the Fermi level.")
    p = eigvecs[:, occupied] @ eigvecs[:, occupied].conj().T

    Lx = positions[:, 0].max() - positions[:, 0].min()
    Ly = positions[:, 1].max() - positions[:, 1].min()

    x = np.diag(positions[:, 0])
    y = np.diag(positions[:, 1])

    ux = expm(2j * np.pi * x / Lx)
    uy = expm(2j * np.pi * y / Ly)

    U = p @ ux @ p
    V = p @ uy @ p
    w = V @ U @ V.conj().T @ U.conj().T

    _, logdet = np.linalg.slogdet(w)
    bott = logdet.imag / (2 * np.pi)
    return bott