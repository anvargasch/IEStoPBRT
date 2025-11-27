# ies2exr.py
# Robust IES → EXR (square map for PBRT goniometric lights)
# Author: Angélica Vargas Chavarro + ChatGPT
# Date: 2025-11-27 (normalized for PBRT "power" = luminous flux)
# License: MIT

import re
import numpy as np
import math
import pyexr   # 👉 ahora usamos pyexr


# -----------------------------
# Utility: save EXR using pyexr
# -----------------------------
def save_exr(filename: str, img: np.ndarray):
    """
    Save a float32 numpy array (H,W) or (H,W,3) as an EXR file using pyexr.
    """
    if img.ndim == 2:  # grayscale
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] != 3:
        raise ValueError("Image must have 1 or 3 channels")

    img = img.astype(np.float32)
    pyexr.write(filename, img)
    print(f"✅ EXR saved to {filename}")


# -----------------------------
# IES Parsing Utilities
# -----------------------------
def load_ies(filename: str):
    with open(filename, "r", encoding="latin-1") as f:
        content = f.read()
    match = re.search(r"TILT=", content)
    if not match:
        raise ValueError("Invalid IES: missing 'TILT='")
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if "TILT=" in line:
            return lines, i
    raise ValueError("Invalid IES: TILT section not found")


def read_header(lines, idx: int):
    """
    Lee la línea numérica principal después de TILT=.

    Formato típico LM-63:
    [num_lamps] [lumens_per_lamp] [candela_mult] [n_vert] [n_horz]
    [photometric_type] [units_type] [width] [length] [height] ...
    """
    vals = lines[idx + 1].split()
    if len(vals) < 7:
        raise ValueError("IES header line seems incomplete")

    num_lamps        = int(vals[0])
    lumens_per_lamp  = float(vals[1])
    candela_mult     = float(vals[2])
    n_vert           = int(vals[3])
    n_horz           = int(vals[4])
    photometric_type = int(vals[5])  # no lo usamos de momento
    units_type       = int(vals[6])  # 1=feet, 2=meters

    return num_lamps, lumens_per_lamp, candela_mult, n_vert, n_horz, units_type


def read_angles(lines, idx: int, count: int):
    angles = []
    while len(angles) < count:
        parts = lines[idx].split()
        angles.extend([float(x) for x in parts])
        idx += 1
    if len(angles) != count:
        raise ValueError("IES: angle count mismatch")
    return np.array(angles, dtype=float), idx


def read_intensity_matrix(lines, idx: int, n_vert: int, n_horz: int):
    values = []
    while len(values) < n_horz * n_vert and idx < len(lines):
        values.extend(lines[idx].split())
        idx += 1
    if len(values) != n_horz * n_vert:
        raise ValueError("IES: intensity count mismatch")
    mat = np.array(values, dtype=float).reshape((n_horz, n_vert))
    return mat.T  # shape (n_vert, n_horz)


# -----------------------------
# Angle Adjustments
# -----------------------------
def adjust_vertical(mat, v_angles, n_vert, n_horz):
    """If vertical angles cover only 0–90°, extend to 180° with zeros."""
    if v_angles[0] == 0 and v_angles[-1] == 90:
        extra = np.linspace(90, 180, n_vert)[1:]
        v_angles = np.concatenate((v_angles, extra))
        zeros = np.zeros((len(extra), n_horz))
        mat = np.concatenate((mat, zeros), axis=0)
    return mat, v_angles


def adjust_horizontal(mat, h_angles, n_horz):
    """Make horizontal coverage full 0–360° by mirroring if needed."""
    if h_angles[0] == 0 and h_angles[-1] == 90:
        # 0–90 → extender a 0–180, luego espejar para 0–360
        extra = np.linspace(90, 180, n_horz)[1:]
        h_angles = np.concatenate((h_angles, extra))
        mirror = mat[:, ::-1][:, 1:]
        mat = np.concatenate((mat, mirror), axis=1)
    elif h_angles[0] == 0 and h_angles[-1] == 180:
        # 0–180 → espejar para 0–360
        extra = np.linspace(180, 360, n_horz)[1:]
        h_angles = np.concatenate((h_angles, extra))
        mirror = mat[:, ::-1][:, 1:]
        mat = np.concatenate((mat, mirror), axis=1)
    return mat, h_angles


# -----------------------------
# Photometric integration
# -----------------------------
def compute_luminous_flux_cd(
    mat_cd: np.ndarray,
    v_angles_deg: np.ndarray,
    h_angles_deg: np.ndarray
) -> float:
    """
    Calcula el flujo luminoso integrando la distribución de intensidad en cd.

    mat_cd:  (n_vert, n_horz) en cd
    v_angles_deg: ángulos verticales en grados (0..180)
    h_angles_deg: ángulos horizontales en grados (0..360)

    Φ ≈ Σ I(θ_i,φ_j) * ΔΩ_ij, con
    ΔΩ_ij = (φ_{j+1}-φ_j)*(cos θ_i - cos θ_{i+1})
    """

    v = np.deg2rad(v_angles_deg)
    h = np.deg2rad(h_angles_deg)

    n_v = len(v)
    n_h = len(h)

    if n_v < 1 or n_h < 1:
        raise ValueError("Empty angle arrays for luminous flux computation")

    # Bordes en vertical (clamp a 0..π)
    v_edges = np.empty(n_v + 1, dtype=float)
    if n_v == 1:
        v_edges[0] = 0.0
        v_edges[1] = math.pi
    else:
        v_edges[1:-1] = 0.5 * (v[:-1] + v[1:])
        v_edges[0] = max(0.0, v[0] - 0.5 * (v[1] - v[0]))
        v_edges[-1] = min(math.pi, v[-1] + 0.5 * (v[-1] - v[-2]))

    # Bordes en horizontal (forzamos 0..2π)
    h_edges = np.empty(n_h + 1, dtype=float)
    if n_h == 1:
        h_edges[0] = 0.0
        h_edges[1] = 2 * math.pi
    else:
        h_edges[1:-1] = 0.5 * (h[:-1] + h[1:])
        h_edges[0] = 0.0
        h_edges[-1] = 2 * math.pi

    dphi = np.diff(h_edges)  # (n_h,)

    phi_total = 0.0
    for i in range(n_v):
        theta0 = v_edges[i]
        theta1 = v_edges[i + 1]
        domega_row = (math.cos(theta0) - math.cos(theta1))
        phi_total += np.sum(mat_cd[i, :] * domega_row * dphi)

    return float(phi_total)  # lumens


def normalize_latlong_image(img: np.ndarray):
    """
    Normaliza un mapa lat-long para que ∫ f(θ,φ) dΩ ≈ 1.

    img: (N,N) con θ ∈ [0,π], φ ∈ [0,2π]
    Return: (img_norm, integral_original)
    """
    H, W = img.shape
    if H != W:
        raise ValueError("Lat-long image should be square (H == W)")
    N = H

    # Centro de cada fila en θ
    theta = math.pi * (np.arange(N) + 0.5) / N  # (N,)
    sin_theta = np.sin(theta)[:, None]          # (N,1)

    dtheta = math.pi / N
    dphi = 2 * math.pi / N

    weights = sin_theta * (dtheta * dphi)      # (N,N) vía broadcasting
    integral = float(np.sum(img * weights))

    if integral <= 0:
        raise ValueError("Non-positive integral for lat-long image normalization")

    img_norm = img / integral
    return img_norm, integral


# -----------------------------
# Mapping to PBRT image (square)
# -----------------------------
def to_pbrt_image(mat, size: int = 512):
    """Resample intensity matrix to a square image (size x size).

    Se asume que la matriz está indexada como [vertical, horizontal].
    """
    import cv2
    resized = cv2.resize(mat, (size, size), interpolation=cv2.INTER_LINEAR)
    # Rotar 180° para mantener la convención que venías usando
    return np.rot90(resized, 2)


# -----------------------------
# Conversion Function
# -----------------------------
def ies_to_exr(filename: str, out_exr: str, size: int = 512):
    """
    Convierte un archivo IES a un EXR cuadrado (size x size) normalizado tal que:
        ∫_sphere tex(θ,φ) dΩ ≈ 1

    Así, en PBRT:
        LightSource "goniometric" ... "float power" [Φ]
    el parámetro 'power' coincide con el flujo luminoso Φ (lm) integrado
    a partir de la distribución IES.
    """
    lines, idx = load_ies(filename)

    (
        num_lamps,
        lumens_per_lamp,
        candela_mult,
        n_vert,
        n_horz,
        units,
    ) = read_header(lines, idx)

    # Bloques de ángulos
    v_angles, idx = read_angles(lines, idx + 3, n_vert)
    h_angles, idx = read_angles(lines, idx, n_horz)

    # Matriz de intensidades
    mat = read_intensity_matrix(lines, idx, n_vert, n_horz)

    # Ajuste de cobertura angular a 0–180 y 0–360
    mat, v_angles = adjust_vertical(mat, v_angles, n_vert, n_horz)
    mat, h_angles = adjust_horizontal(mat, h_angles, n_horz)

    # Aplicar factor de candelas -> matriz en cd
    mat_cd = mat * candela_mult

    # Flujo luminoso integrado a partir de la distribución en cd
    phi_lum_ies = compute_luminous_flux_cd(mat_cd, v_angles, h_angles)

    # Flujo de lámparas desde cabecera (informativo)
    phi_lamps_header = num_lamps * lumens_per_lamp

    # 1) Pasamos la matriz de intensidades en cd al mapa NxN
    img_cd = to_pbrt_image(mat_cd, size=size)   # (N,N) en cd (aprox)

    # 2) Normalizamos como lat-long: integral ≈ 1
    img_norm, integral_before = normalize_latlong_image(img_cd)

    # Asegurar float32 y convertir a RGB
    exr = np.repeat(img_norm[:, :, None], 3, axis=2).astype(np.float32)

    # Guardar EXR con pyexr
    save_exr(out_exr, exr)

    return {
        "num_lamps": num_lamps,
        "lumens_per_lamp_header": lumens_per_lamp,
        "candela_mult": candela_mult,
        "lumens_from_ies_integration": phi_lum_ies,
        "lumens_from_header_lamps": phi_lamps_header,
        "units_type": units,      # 1=ft, 2=m
        "v_angles_count": v_angles.shape[0],
        "h_angles_count": h_angles.shape[0],
        "image_size": size,
        "latlong_integral_before_norm": integral_before,
        "out_exr": out_exr,
        # Valor de 'power' recomendado en PBRT:
        "recommended_power_lm": phi_lum_ies,
    }


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    meta = ies_to_exr(
        "IESfiles/IESsiemens.ies",
        "IESsiemens_square.exr",
        size=1024
    )
    print(meta)
    print(f"Sugerencia PBRT -> float power = {meta['recommended_power_lm']:.2f}")
