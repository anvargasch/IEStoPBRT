import argparse
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pyexr


# ============================================================
# DATA STRUCTURE
# ============================================================
@dataclass
class IESData:
    v_angles_deg: np.ndarray   # (nV,)
    h_angles_deg: np.ndarray   # (nH,)
    cd: np.ndarray             # (nV, nH)
    photometric_type: int
    units_type: int


# ============================================================
# IES PARSING (LM-63, Type C)
# ============================================================
def _find_tilt_index(lines):
    for i, line in enumerate(lines):
        if "TILT=" in line.upper():
            return i
    raise ValueError("IES inválido: no se encontró TILT=")


def _read_floats(lines, start, count):
    vals = []
    i = start
    while len(vals) < count and i < len(lines):
        vals.extend([float(x) for x in lines[i].split()])
        i += 1
    if len(vals) != count:
        raise ValueError("IES inválido: número incorrecto de valores")
    return np.array(vals, float), i


def load_ies_type_c(path: str) -> IESData:
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()

    t = _find_tilt_index(lines)
    tilt = lines[t].split("=")[1].strip().upper()
    if tilt != "NONE":
        raise ValueError("Solo se admite TILT=NONE")

    header = lines[t + 1].split()
    if len(header) < 7:
        raise ValueError("IES inválido: cabecera incompleta")

    candela_mult = float(header[2])
    n_vert = int(header[3])
    n_horz = int(header[4])
    photometric_type = int(header[5])
    units_type = int(header[6])

    # localizar bloques de ángulos (robusto)
    start = t + 2
    for k in range(6):
        try:
            v_angles, i1 = _read_floats(lines, start + k, n_vert)
            h_angles, i2 = _read_floats(lines, i1, n_horz)
            break
        except Exception:
            continue
    else:
        raise ValueError("No se pudieron leer ángulos")

    # leer intensidades (por planos horizontales)
    needed = n_vert * n_horz
    vals = []
    i = i2
    while len(vals) < needed:
        vals.extend(lines[i].split())
        i += 1

    cd = np.array(vals, float).reshape((n_horz, n_vert)).T
    cd *= candela_mult

    # ordenar por seguridad
    v_idx = np.argsort(v_angles)
    h_idx = np.argsort(h_angles)
    v_angles = v_angles[v_idx]
    h_angles = h_angles[h_idx]
    cd = cd[v_idx][:, h_idx]

    return IESData(v_angles, h_angles, cd, photometric_type, units_type)


# ============================================================
# ANGULAR COMPLETION (TYPE C)
# ============================================================
def complete_horizontal(cd, h):
    hmax = h[-1]

    if abs(hmax - 360) < 1e-6:
        return cd[:, :-1], h[:-1] if h[-1] == 360 else (cd, h)

    if abs(hmax - 180) < 1e-6:
        h2 = 360 - h[1:-1][::-1]
        cd2 = cd[:, 1:-1][:, ::-1]
        return np.hstack([cd, cd2]), np.concatenate([h, h2])

    if abs(hmax - 90) < 1e-6:
        h180 = np.concatenate([h, 180 - h[1:-1][::-1]])
        cd180 = np.hstack([cd, cd[:, 1:-1][:, ::-1]])
        h360 = np.concatenate([h180, 360 - h180[1:-1][::-1]])
        cd360 = np.hstack([cd180, cd180[:, 1:-1][:, ::-1]])
        return cd360, h360

    raise ValueError("Rango horizontal no soportado")


def ensure_vertical(cd, v):
    if v[-1] <= 90 + 1e-6:
        step = np.median(np.diff(v))
        v_ext = np.arange(v[-1] + step, 180 + 1e-6, step)
        cd_ext = np.zeros((len(v_ext), cd.shape[1]))
        return np.vstack([cd, cd_ext]), np.concatenate([v, v_ext])
    return cd, v


# ============================================================
# IRREGULAR ANGULAR INTERPOLATION
# ============================================================
def sample_irregular(cd, v, h, theta, phi):
    phi = np.mod(phi, 360)
    h_ext = np.append(h, h[0] + 360)
    cd_ext = np.hstack([cd, cd[:, :1]])

    i1 = np.searchsorted(v, theta, "right")
    i0 = np.clip(i1 - 1, 0, len(v) - 1)
    i1 = np.clip(i1, 0, len(v) - 1)

    j1 = np.searchsorted(h_ext, phi, "right")
    j0 = np.clip(j1 - 1, 0, len(h_ext) - 2)
    j1 = np.clip(j1, 1, len(h_ext) - 1)

    tv = np.where(v[i1] > v[i0], (theta - v[i0]) / (v[i1] - v[i0]), 0)
    tp = np.where(h_ext[j1] > h_ext[j0], (phi - h_ext[j0]) / (h_ext[j1] - h_ext[j0]), 0)

    I = (
        (1-tv)*(1-tp)*cd_ext[i0, j0] +
        (1-tv)*tp*cd_ext[i0, j1] +
        tv*(1-tp)*cd_ext[i1, j0] +
        tv*tp*cd_ext[i1, j1]
    )
    return I


# ============================================================
# BUILD LAT-LONG + NORMALIZATION
# ============================================================
def build_latlong_normalized(cd, v, h, size):
    N = size
    theta = 180 * (np.arange(N) + 0.5) / N
    phi = 360 * (np.arange(N) + 0.5) / N
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    img = sample_irregular(cd, v, h, TH, PH).astype(float)

    theta_r = math.pi * (np.arange(N) + 0.5) / N
    weights = np.sin(theta_r)[:, None] * (math.pi/N) * (2*math.pi/N)
    img /= np.sum(img * weights)
    return img


def save_exr(path, img):
    rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    pyexr.write(path, rgb)


# ============================================================
# PUBLIC API (NOTEBOOK)
# ============================================================
def ies_to_exr(filename, output, size=1024):
    ies = load_ies_type_c(filename)
    cd, v = ensure_vertical(ies.cd, ies.v_angles_deg)
    cd, h = complete_horizontal(cd, ies.h_angles_deg)
    img = build_latlong_normalized(cd, v, h, size)
    save_exr(output, img)
    return output


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ies")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-s", "--size", type=int, default=1024)
    args = ap.parse_args()

    ies_to_exr(args.ies, args.out, args.size)
    print(f"OK: {args.out}")


if __name__ == "__main__":
    main()
