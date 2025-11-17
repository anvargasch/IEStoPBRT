# ies2exr.py
# Robust IES → EXR (square map for PBRT goniometric lights)
# Author: Angélica Vargas Chavarro + ChatGPT
# Date: 2025-11-17 (updated to generate square EXR)
# License: MIT

import re
import numpy as np
import math
import OpenEXR, Imath   # EXR saving

# -----------------------------
# Utility: save EXR using OpenEXR
# -----------------------------
def save_exr(filename: str, img: np.ndarray):
    """
    Save a float32 numpy array (H,W) or (H,W,3) as an EXR file using OpenEXR.
    
    Parameters
    ----------
    filename : str
        Output EXR file path
    img : np.ndarray
        Image array. If shape is (H,W) → replicate channels into RGB.
                   If shape is (H,W,3) → treated as RGB.
    """
    if img.ndim == 2:  # grayscale
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] != 3:
        raise ValueError("Image must have 1 or 3 channels")

    height, width, _ = img.shape

    # EXR header
    header = OpenEXR.Header(width, height)
    pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {
        'R': Imath.Channel(pix_type),
        'G': Imath.Channel(pix_type),
        'B': Imath.Channel(pix_type),
    }

    # Channel planes
    r = img[:, :, 0].astype(np.float32).tobytes()
    g = img[:, :, 1].astype(np.float32).tobytes()
    b = img[:, :, 2].astype(np.float32).tobytes()

    # Write file
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({'R': r, 'G': g, 'B': b})
    out.close()
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
    vals = lines[idx + 1].split()
    lumens_per_lamp = float(vals[1])
    candela_mult = float(vals[2])
    n_vert = int(vals[3])
    n_horz = int(vals[4])
    units_type = int(vals[6])  # 1=feet, 2=meters
    return lumens_per_lamp, candela_mult, n_vert, n_horz, units_type


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
        # 0–90 → extend to 0–180, then mirror to complete 0–360
        extra = np.linspace(90, 180, n_horz)[1:]
        h_angles = np.concatenate((h_angles, extra))
        mirror = mat[:, ::-1][:, 1:]
        mat = np.concatenate((mat, mirror), axis=1)
    elif h_angles[0] == 0 and h_angles[-1] == 180:
        # 0–180 → mirror to 0–360
        extra = np.linspace(180, 360, n_horz)[1:]
        h_angles = np.concatenate((h_angles, extra))
        mirror = mat[:, ::-1][:, 1:]
        mat = np.concatenate((mat, mirror), axis=1)
    return mat, h_angles


# -----------------------------
# Mapping to PBRT image (square)
# -----------------------------
def to_pbrt_image(mat, size: int = 512):
    """Resample intensity matrix to a square image (size x size).

    The original matrix is indexed as [vertical, horizontal]. Here we
    map vertical and horizontal angles linearly to the vertical and
    horizontal pixel indices, respectively, and then rotate 180° so
    that the "up" direction is consistent with PBRT's conventions.
    """
    import cv2
    # cv2.resize expects (width, height)
    resized = cv2.resize(mat, (size, size), interpolation=cv2.INTER_LINEAR)
    # Rotate 180° to match PBRT goniometric map convention
    return np.rot90(resized, 2)


# -----------------------------
# Conversion Function
# -----------------------------
def ies_to_exr(filename: str, out_exr: str, size: int = 512):
    """Convert an IES file to a square EXR (size x size)."""
    lines, idx = load_ies(filename)
    lumens, factor, n_vert, n_horz, units = read_header(lines, idx)

    # Angle blocks
    v_angles, idx = read_angles(lines, idx + 3, n_vert)
    h_angles, idx = read_angles(lines, idx, n_horz)

    # Intensity matrix
    mat = read_intensity_matrix(lines, idx, n_vert, n_horz)

    # Adjust coverage to 0–180 (vertical) and 0–360 (horizontal)
    mat, v_angles = adjust_vertical(mat, v_angles, n_vert, n_horz)
    mat, h_angles = adjust_horizontal(mat, h_angles, n_horz)

    # Generate square image
    img = to_pbrt_image(mat, size=size)

    # Ensure 3 channels float32
    exr = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)

    # Save with OpenEXR
    save_exr(out_exr, exr)

    return {
        "lumens": lumens,
        "factor": factor,
        "units": units,
        "v_angles": v_angles.shape[0],
        "h_angles": h_angles.shape[0],
        "size": size,
        "out_exr": out_exr,
    }


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    # This will now generate a 1024 x 1024 EXR
    meta = ies_to_exr("IESfiles/IESsiemens.ies", "IESsiemens_square.exr", size=1024)
    print(meta)
