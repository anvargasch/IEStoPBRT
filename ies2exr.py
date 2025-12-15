import argparse
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pyexr


# =========================
# IES parsing (LM-63)
# =========================
@dataclass
class IESData:
    v_angles_deg: np.ndarray          # (nV,)
    h_angles_deg: np.ndarray          # (nH,) (tal como venga en el archivo)
    cd: np.ndarray                    # (nV, nH) candelas
    photometric_type: int
    units_type: int


def _find_tilt_index(lines):
    for i, line in enumerate(lines):
        if "TILT=" in line.upper():
            return i
    raise ValueError("IES inválido: no se encontró 'TILT='.")


def _read_floats_from_lines(lines, start_idx, count) -> Tuple[np.ndarray, int]:
    vals = []
    idx = start_idx
    while len(vals) < count and idx < len(lines):
        parts = lines[idx].strip().split()
        if parts:
            vals.extend([float(p) for p in parts])
        idx += 1
    if len(vals) != count:
        raise ValueError(f"IES inválido: se esperaban {count} valores, llegaron {len(vals)}.")
    return np.array(vals, dtype=float), idx


def _read_matrix_from_lines(lines, start_idx, n_rows, n_cols) -> Tuple[np.ndarray, int]:
    # En IES, los datos suelen venir por planos horizontales (nH) con nV valores cada uno.
    # Nosotros lo reordenamos a (nV, nH).
    needed = n_rows * n_cols
    vals = []
    idx = start_idx
    while len(vals) < needed and idx < len(lines):
        parts = lines[idx].strip().split()
        if parts:
            vals.extend([float(p) for p in parts])
        idx += 1
    if len(vals) != needed:
        raise ValueError(f"IES inválido: se esperaban {needed} intensidades, llegaron {len(vals)}.")
    mat = np.array(vals, dtype=float).reshape((n_rows, n_cols))
    return mat, idx


def load_ies_type_c(path: str) -> IESData:
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()

    t_idx = _find_tilt_index(lines)
    tilt_line = lines[t_idx].strip().upper()

    if tilt_line.startswith("TILT="):
        tilt_mode = tilt_line.split("=", 1)[1].strip()
    else:
        tilt_mode = "NONE"

    # Leemos línea numérica principal (después de TILT=...)
    header_tokens = lines[t_idx + 1].split()
    if len(header_tokens) < 7:
        raise ValueError("IES inválido: cabecera numérica incompleta después de TILT=.")

    num_lamps = int(header_tokens[0])
    lumens_per_lamp = float(header_tokens[1])  # no lo usamos aquí
    candela_mult = float(header_tokens[2])
    n_vert = int(header_tokens[3])
    n_horz = int(header_tokens[4])
    photometric_type = int(header_tokens[5])   # 1=Type C (en muchos LM-63)
    units_type = int(header_tokens[6])         # 1=feet, 2=meters

    # Saltar bloque TILT si no es NONE
    # Para evitar “mugre” y comportamientos inesperados, sólo soportamos NONE de forma estricta.
    if tilt_mode not in ("NONE", "NO", "NONE\r"):
        raise ValueError(
            f"TILT={tilt_mode} no soportado en esta versión final. "
            "Exporta el IES con TILT=NONE o aplícalo en el software fotométrico antes."
        )

    # En LM-63, después de la cabecera suelen venir 3 números adicionales (ballast/future use/watts)
    # Muchas veces están en la línea t_idx+2, pero no siempre; se manejan consumiendo el bloque de ángulos.
    # Usamos el layout clásico: v_angles empiezan en t_idx+3.
    idx = t_idx + 2
    # Consumir posibles líneas numéricas “extra” hasta que empecemos a leer ángulos.
    # (Estrategia robusta: leemos v_angles con el contador exacto desde idx+1)
    idx = t_idx + 2

    # Heurística robusta:
    # buscamos el punto donde podamos leer n_vert floats seguidos; asumimos que es la línea siguiente a los "3 extras".
    # En la práctica, suele ser t_idx+3. Si falla, retrocedemos/avanzamos.
    start_try = t_idx + 2
    found = None
    for k in range(6):  # prueba unas pocas líneas hacia adelante
        try:
            v_angles, idx_after_v = _read_floats_from_lines(lines, start_try + k, n_vert)
            h_angles, idx_after_h = _read_floats_from_lines(lines, idx_after_v, n_horz)
            # si esto funcionó, asumimos correcto
            found = (v_angles, h_angles, idx_after_h)
            break
        except Exception:
            continue
    if found is None:
        raise ValueError("No fue posible localizar los bloques de ángulos (vertical/horizontal).")

    v_angles, h_angles, idx = found

    # Intensidades
    mat_hv, idx2 = _read_matrix_from_lines(lines, idx, n_horz, n_vert)  # (nH, nV)
    mat_vh = mat_hv.T                                                   # (nV, nH)
    cd = mat_vh * float(candela_mult)

    # Orden ascendente por seguridad (si el archivo viniera desordenado)
    v_sort = np.argsort(v_angles)
    v_angles = v_angles[v_sort]
    cd = cd[v_sort, :]

    h_sort = np.argsort(h_angles)
    h_angles = h_angles[h_sort]
    cd = cd[:, h_sort]

    return IESData(
        v_angles_deg=v_angles.astype(float),
        h_angles_deg=h_angles.astype(float),
        cd=cd.astype(float),
        photometric_type=photometric_type,
        units_type=units_type
    )


# =========================
# Angular completion (Type C)
# =========================
def _complete_horizontal_0_360(cd: np.ndarray, h_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Completa h a [0,360) con simetrías típicas (0..90, 0..180, 0..360).
    Asume h inicia en 0.
    """
    h = np.array(h_deg, dtype=float)
    if abs(h[0] - 0.0) > 1e-6:
        raise ValueError("Se requiere que los ángulos horizontales inicien en 0° para completar simetrías.")

    hmax = float(h[-1])

    # Caso 0..360 (con o sin 360 repetido)
    if abs(hmax - 360.0) < 1e-6:
        # si incluye 360 como duplicado de 0, lo removemos para interpolación periódica limpia
        if len(h) > 1 and abs(h[-1] - 360.0) < 1e-6:
            h = h[:-1]
            cd = cd[:, :-1]
        return cd, h

    # Caso 0..180 -> espejo a 360
    if abs(hmax - 180.0) < 1e-6:
        h_inner = h[1:-1]
        h_mirr = 360.0 - h_inner[::-1]
        cd_inner = cd[:, 1:-1]
        cd_mirr = cd_inner[:, ::-1]
        h_full = np.concatenate([h, h_mirr])
        cd_full = np.concatenate([cd, cd_mirr], axis=1)
        return cd_full, h_full

    # Caso 0..90 -> espejo a 180 y luego a 360
    if abs(hmax - 90.0) < 1e-6:
        # 0..90 -> 0..180
        h_inner = h[1:-1]
        h_180_mirr = 180.0 - h_inner[::-1]
        cd_inner = cd[:, 1:-1]
        cd_180_mirr = cd_inner[:, ::-1]
        h_180 = np.concatenate([h, h_180_mirr])
        cd_180 = np.concatenate([cd, cd_180_mirr], axis=1)

        # 0..180 -> 0..360
        h_inner2 = h_180[1:-1]
        h_360_mirr = 360.0 - h_inner2[::-1]
        cd_inner2 = cd_180[:, 1:-1]
        cd_360_mirr = cd_inner2[:, ::-1]
        h_full = np.concatenate([h_180, h_360_mirr])
        cd_full = np.concatenate([cd_180, cd_360_mirr], axis=1)
        return cd_full, h_full

    raise ValueError(f"Rango horizontal no soportado automáticamente: 0..{hmax}°.")


def _ensure_vertical_0_180(cd: np.ndarray, v_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asegura soporte vertical 0..180:
    - Si llega 0..90: extiende 90..180 con ceros.
    - Si llega 0..180: deja igual.
    """
    v = np.array(v_deg, dtype=float)
    vmax = float(v[-1])

    if vmax <= 90.0 + 1e-6:
        # Extender a 180 con ceros, manteniendo resolución similar (paso medio)
        if len(v) >= 2:
            step = float(np.median(np.diff(v)))
            step = step if step > 0 else 1.0
        else:
            step = 1.0
        v_ext = np.arange(vmax + step, 180.0 + 1e-6, step, dtype=float)
        if v_ext.size == 0:
            # ya está en 90 exacto sin más
            v_full = v
            cd_full = cd
        else:
            cd_zeros = np.zeros((cd.shape[0] + v_ext.size, cd.shape[1]), dtype=float)
            cd_zeros[: cd.shape[0], :] = cd
            v_full = np.concatenate([v, v_ext])
            cd_full = cd_zeros
        return cd_full, v_full

    if abs(vmax - 180.0) < 1e-6 or vmax < 180.0 + 1e-6:
        return cd, v

    raise ValueError(f"Ángulos verticales fuera de rango esperado: max={vmax}°.")


# =========================
# Irregular bilinear interpolation in (theta, phi)
# =========================
def _sample_cd_irregular_bilinear(cd_vh: np.ndarray,
                                  v_deg: np.ndarray,
                                  h_deg_0_360: np.ndarray,
                                  theta_deg: np.ndarray,
                                  phi_deg: np.ndarray) -> np.ndarray:
    """
    Interpola bilinealmente en una grilla irregular (v,h).
    - v: 0..180
    - h: 0..360 (sin 360 repetido)
    - phi se trata periódico.
    """
    v = np.array(v_deg, dtype=float)
    h = np.array(h_deg_0_360, dtype=float)

    # wrap phi a [0,360)
    phi = np.mod(phi_deg, 360.0)
    theta = np.array(theta_deg, dtype=float)

    # Cierre periódico en h: añadimos columna extra en 360 igual a 0
    h_ext = np.concatenate([h, [h[0] + 360.0]])
    cd_ext = np.concatenate([cd_vh, cd_vh[:, 0:1]], axis=1)

    # índices theta
    i1 = np.searchsorted(v, theta, side="right")
    i0 = np.clip(i1 - 1, 0, len(v) - 1)
    i1 = np.clip(i1,     0, len(v) - 1)
    v0 = v[i0]
    v1 = v[i1]
    tv = np.where(v1 > v0, (theta - v0) / (v1 - v0), 0.0)

    # índices phi
    j1 = np.searchsorted(h_ext, phi, side="right")
    j0 = np.clip(j1 - 1, 0, len(h_ext) - 2)
    j1 = np.clip(j1,     1, len(h_ext) - 1)
    h0 = h_ext[j0]
    h1 = h_ext[j1]
    tp = np.where(h1 > h0, (phi - h0) / (h1 - h0), 0.0)

    I00 = cd_ext[i0, j0]
    I01 = cd_ext[i0, j1]
    I10 = cd_ext[i1, j0]
    I11 = cd_ext[i1, j1]

    I0 = I00 * (1 - tp) + I01 * tp
    I1 = I10 * (1 - tp) + I11 * tp
    I = I0 * (1 - tv) + I1 * tv
    return I


# =========================
# Lat-long build + solid-angle normalization
# =========================
def build_latlong_normalized(cd_vh: np.ndarray, v_deg: np.ndarray, h_deg: np.ndarray, size: int) -> np.ndarray:
    """
    Construye imagen lat-long (H=W=size):
    - filas: theta 0..180
    - columnas: phi 0..360
    Luego normaliza para que ∫ f dΩ = 1.
    """
    N = int(size)
    theta = 180.0 * (np.arange(N) + 0.5) / N  # (N,)
    phi = 360.0 * (np.arange(N) + 0.5) / N    # (N,)

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    img = _sample_cd_irregular_bilinear(cd_vh, v_deg, h_deg, TH, PH).astype(np.float64)

    # Normalización por ángulo sólido
    theta_r = math.pi * (np.arange(N) + 0.5) / N
    sin_theta = np.sin(theta_r)[:, None]
    dtheta = math.pi / N
    dphi = 2 * math.pi / N
    weights = sin_theta * (dtheta * dphi)
    integral = float(np.sum(img * weights))
    if integral <= 0:
        raise ValueError("No se pudo normalizar: integral no positiva.")
    img /= integral
    return img


def save_exr_gray_as_rgb(path: str, img: np.ndarray):
    if img.ndim != 2:
        raise ValueError("Se esperaba imagen 2D (grayscale).")
    rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    pyexr.write(path, rgb)


# =========================
# Optional: call PBRT imgtool makeequiarea
# =========================
def try_makeequiarea(imgtool: str, latlong_exr: str, out_ea_exr: str):
    cmd = [imgtool, "makeequiarea", latlong_exr, out_ea_exr]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"imgtool makeequiarea falló:\n{p.stderr.strip()}")
    return True


# =========================
# Main
# =========================
def ies2exr(ies_path: str,
                       out_exr: str,
                       size: int = 1024,
                       also_write_latlong: bool = False,
                       imgtool_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    ies = load_ies_type_c(ies_path)

    cd, v = _ensure_vertical_0_180(ies.cd, ies.v_angles_deg)
    cd, h = _complete_horizontal_0_360(cd, ies.h_angles_deg)

    img_latlong = build_latlong_normalized(cd, v, h, size=size)

    latlong_path = None
    if also_write_latlong or (imgtool_path is not None):
        latlong_path = os.path.splitext(out_exr)[0] + "_latlong.exr"
        save_exr_gray_as_rgb(latlong_path, img_latlong)

    # Si hay imgtool, convertimos a octahedral equal-area para PBRT v4
    if imgtool_path is not None:
        # si no se escribió latlong todavía, se escribe ahora
        if latlong_path is None:
            latlong_path = os.path.splitext(out_exr)[0] + "_latlong.exr"
            save_exr_gray_as_rgb(latlong_path, img_latlong)

        try_makeequiarea(imgtool_path, latlong_path, out_exr)
    else:
        # Sin imgtool, dejamos latlong como salida (si out_exr apunta directo)
        # Para evitar confusión, si no hay imgtool y no se pidió latlong, igual escribimos out_exr como latlong.
        save_exr_gray_as_rgb(out_exr, img_latlong)

    return out_exr, latlong_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ies", help="Archivo .ies (Type C)")
    ap.add_argument("-o", "--out", required=True, help="EXR de salida (idealmente EA si usas imgtool)")
    ap.add_argument("-s", "--size", type=int, default=1024, help="Resolución (NxN), recomendado 1024")
    ap.add_argument("--imgtool", default=None, help="Ruta a imgtool (PBRT). Si se da, genera EXR octahedral EA.")
    ap.add_argument("--keep-latlong", action="store_true", help="Guardar también el latlong intermedio")
    args = ap.parse_args()

    out, latlong = ies2exr(
        ies_path=args.ies,
        out_exr=args.out,
        size=args.size,
        also_write_latlong=args.keep_latlong,
        imgtool_path=args.imgtool
    )

    print(f"OK: {out}")
    if latlong:
        print(f"LatLong: {latlong}")


if __name__ == "__main__":
    main()


