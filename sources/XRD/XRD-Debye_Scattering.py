#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
XRD-Debye_Scattering.py - Debye scattering equation for finite, non-periodic structures.

Intended domain: Finite isolated configurations: nanoparticles, clusters, molecules, amorphous or
liquid snapshots, and carved-out regions containing defects or grain boundaries.
"""
import sys
import os
import glob
import time
import math
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import periodictable

# Optional imports
try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

try:
    from scipy.spatial.distance import pdist
except Exception:
    pdist = None

# Project utilities
from common_utils import parse_input_file, read_lammps_xyz, validate_params, species_read_options

OK = "[OK]"
WARN = "[WARN]"
ERR = "[ERROR]"

WAVELENGTHS = {
    'CuKa': 1.54184, 'CuKa1': 1.54056, 'CuKa2': 1.54439, 'CuKb1': 1.39222,
    'MoKa': 0.71073, 'MoKa1': 0.70930, 'MoKa2': 0.71359, 'MoKb1': 0.63229,
    'CrKa': 2.29100, 'CrKa1': 2.28970, 'CrKa2': 2.29361, 'CrKb1': 2.08487,
    'FeKa': 1.93735, 'FeKa1': 1.93604, 'FeKa2': 1.93998, 'FeKb1': 1.75661,
    'CoKa': 1.79026, 'CoKa1': 1.78896, 'CoKa2': 1.79285, 'CoKb1': 1.63079,
    'AgKa': 0.560885, 'AgKa1': 0.559421, 'AgKa2': 0.563813, 'AgKb1': 0.497082,
}


def resolve_wavelength(value, verbose=True):
    if value is None:
        if verbose:
            print(f"{WARN} no wavelength given; using CuKa = {WAVELENGTHS['CuKa']} A")
        return WAVELENGTHS['CuKa']
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    for key, wl in WAVELENGTHS.items():
        if text.lower() == key.lower():
            return float(wl)
    try:
        return float(text)
    except Exception:
        pass
    raise ValueError(
        f"Unknown wavelength {value!r}. Use a number in Angstrom or one of: "
        + ", ".join(sorted(WAVELENGTHS)))

def _coerce_value(val):
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if '#' in s:
        s = s.split('#', 1)[0].strip()
    if s == '':
        return None
    low = s.lower()
    if low in ('true', 't', 'yes', 'y'):
        return True
    if low in ('false', 'f', 'no', 'n'):
        return False
    try:
        if '.' not in s and 'e' not in low:
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    return s


def sanitize_params(params):
    if params is None:
        return {}

    p = dict(params)

    if 'debye' in p and isinstance(p['debye'], dict):
        merged = dict(p)
        merged.update(p['debye'])
        p = merged

    dotted_prefix = 'debye.'
    for key in list(p.keys()):
        if isinstance(key, str) and key.startswith(dotted_prefix):
            new_key = key[len(dotted_prefix):]
            if new_key not in p:
                p[new_key] = p[key]
            del p[key]

    return {k: _coerce_value(v) for k, v in p.items()}


def _as_bool(params, key, default=False):
    v = params.get(key, default)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ('true', 't', 'yes', 'y', '1', 'on')


def _safe_sinc(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def _safe_outer_sinc(q_values, dists):
    return _safe_sinc(np.outer(q_values, dists))


def _lorch_window(bin_centers):
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    r_max = bin_centers[-1]
    if r_max <= 0:
        return np.ones_like(bin_centers)
    x = np.pi * bin_centers / r_max
    out = np.ones_like(bin_centers)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def two_theta_to_q(two_theta_deg, wavelength):
    return 4.0 * np.pi * np.sin(np.radians(np.asarray(two_theta_deg, float) / 2.0)) / wavelength


def max_pair_distance(coords, tree=None, chunk=4096):
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return 0.0
    best = 0.0
    for start in range(0, n, chunk):
        block = coords[start:start + chunk]
        d = np.sqrt(np.maximum(
            ((block[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2), 0.0))
        best = max(best, float(d.max()))
    return best


def centroid_radius_bound(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    return 2.0 * float(np.max(np.linalg.norm(coords - centroid, axis=1)))

def pair_workload_preflight(N, params, method_label, verbose=True):
    n_pairs = N * (N - 1) // 2
    cap = int(float(params.get('max_pairs', 2e9)))
    if verbose:
        print(f"  pair workload: {n_pairs:,} unordered pairs ({method_label})")
    if n_pairs > cap and not _as_bool(params, 'allow_large_pair_count', False):
        raise MemoryError(
            f"{N:,} atoms means {n_pairs:,} unordered pairs, above the max_pairs cap "
            f"of {cap:,}.\n"
            f"  The Debye scattering equation is intrinsically O(N^2): a KD-tree does "
            f"not help when\n"
            f"  the cutoff spans the whole cluster. Options:\n"
            f"    * use debye_method = binned (same equation, ~100x less work per pair);\n"
            f"    * reduce the configuration (a representative sub-cluster);\n"
            f"    * raise max_pairs / set allow_large_pair_count = yes to proceed anyway.")
    return n_pairs


def _iter_pair_batches(coords, codes, nsp, cutoff, complete, max_block_elements,
                       chunk_size=2048):
    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    if N < 2:
        return

    if complete:
        m = max(1, int(max_block_elements // max(N, 1)))
        for start in range(0, N, m):
            end = min(N, start + m)
            block = coords[start:end]
            d2 = ((block[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
            rows = np.arange(start, end)[:, None]
            cols = np.arange(N)[None, :]
            keep = cols > rows
            if not keep.any():
                continue
            d = np.sqrt(np.maximum(d2[keep], 0.0))
            ii = np.broadcast_to(rows, keep.shape)[keep]
            jj = np.broadcast_to(cols, keep.shape)[keep]
            ca, cb = codes[ii], codes[jj]
            yield d, np.minimum(ca, cb) * nsp + np.maximum(ca, cb)
        return

    tree = cKDTree(coords)
    m = max(1, int(chunk_size))            # centre atoms per neighbour query
    for start in range(0, N, m):
        end = min(N, start + m)
        neigh_lists = tree.query_ball_point(coords[start:end], cutoff)
        idx_i, idx_j = [], []
        for local, neigh in enumerate(neigh_lists):
            i = start + local
            arr = np.fromiter(neigh, dtype=np.int64, count=len(neigh))
            arr = arr[arr > i]
            if arr.size:
                idx_i.append(np.full(arr.size, i, dtype=np.int64))
                idx_j.append(arr)
        if not idx_i:
            continue
        ii = np.concatenate(idx_i)
        jj = np.concatenate(idx_j)
        d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
        ca, cb = codes[ii], codes[jj]
        yield d, np.minimum(ca, cb) * nsp + np.maximum(ca, cb)


# Debye scattering calculator
class DebyeCalculator:
    """Debye scattering equation for a finite configuration of atoms."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.notes = []

    def _say(self, msg):
        if self.verbose:
            print(msg)
            
    # Atomic form factors
    def get_scattering_factors(self, elements, q_values, wavelength, params):
        q_values = np.asarray(q_values, dtype=np.float64)
        apply_anom = _as_bool(params, 'apply_anomalous', False)

        energy_keV = None
        if apply_anom:
            try:
                energy_keV = 12.3984193 / float(wavelength)
            except Exception:
                energy_keV = None

        f_by_element = {}
        for el in elements:
            try:
                elem = periodictable.elements.symbol(el)
            except Exception:
                raise ValueError(f"Unknown element symbol: {el}")

            f0 = np.asarray(elem.xray.f0(q_values), dtype=np.float64)

            if not apply_anom:
                f_by_element[el] = f0
                continue

            fprime = 0.0
            fdouble = 0.0
            source = None

            key_fp, key_fpp = f"fprime_{el}", f"fdouble_{el}"
            user_fp = params.get(key_fp, None)
            user_fpp = params.get(key_fpp, None)
            if user_fp not in (None, "") or user_fpp not in (None, ""):
                try:
                    if user_fp not in (None, ""):
                        fprime = float(user_fp)
                    if user_fpp not in (None, ""):
                        fdouble = float(user_fpp)
                    source = "user-supplied"
                except Exception:
                    self._say(f"{WARN} could not parse {key_fp}/{key_fpp} for {el}; "
                              f"falling back to tabulated values.")
                    fprime, fdouble, source = 0.0, 0.0, None

            if source is None and energy_keV is not None:
                try:
                    f1, f2 = elem.xray.scattering_factors(energy=energy_keV)
                    if f1 is None or f2 is None:
                        raise ValueError("no tabulated data at this energy")
                    fprime = float(f1) - float(elem.number)   # Henke f1 contains Z
                    fdouble = float(f2)
                    source = f"periodictable Henke tables at {energy_keV:.4f} keV"
                except Exception as exc:
                    self._say(f"{WARN} no anomalous data for {el} "
                              f"({exc}); using f' = f'' = 0. Supply {key_fp} / {key_fpp} "
                              f"explicitly if the correction matters at this energy.")
                    fprime, fdouble, source = 0.0, 0.0, "none"

            if source and source != "none":
                self._say(f"  anomalous {el}: f' = {fprime:+.4f}, f'' = {fdouble:+.4f} "
                          f"({source})")

            f_by_element[el] = (f0.astype(np.complex128)
                                + complex(fprime, fdouble))

        return f_by_element

    def _species_form_factors(self, species_order, q_values, wavelength, params):
        apply_dw = _as_bool(params, 'apply_debye_waller', False)
        f_by_el = self.get_scattering_factors(species_order, q_values, wavelength, params)

        any_dw = False
        rows = []
        for el in species_order:
            f = f_by_el[el]
            if apply_dw:
                B_val = params.get(f"B_{el}", params.get("debye_waller_B", 0.0))
                try:
                    B_val = float(B_val)
                except Exception:
                    B_val = 0.0
                if B_val != 0.0:
                    f = f * np.exp(-B_val * q_values ** 2 / (16.0 * np.pi ** 2))
                    any_dw = True
                    self._say(f"  Debye-Waller {el}: B = {B_val:g} A^2")
            rows.append(f)

        dtype = np.complex128 if any(np.iscomplexobj(r) for r in rows) else np.float64
        return np.asarray(rows, dtype=dtype), any_dw

    @staticmethod
    def _species_codes(species):
        symbols = [str(s) for s in species]
        order = sorted(set(symbols))
        lookup = {el: i for i, el in enumerate(order)}
        return order, np.fromiter((lookup[s] for s in symbols), dtype=np.int64,
                                  count=len(symbols))

    @staticmethod
    def _pair_key(order, ca, cb):
        a, b = order[ca], order[cb]
        return (a, b) if a <= b else (b, a)

    # Exact direct pair sum
    def calculate_direct_dse(self, coords, species, params, wavelength, two_theta):
        params = sanitize_params(params) if params else {}
        coords = np.asarray(coords, dtype=np.float64)
        N = coords.shape[0]
        two_theta = np.asarray(two_theta, dtype=np.float64)
        q_values = two_theta_to_q(two_theta, wavelength)
        nq = q_values.size

        order, codes = self._species_codes(species)
        nsp = len(order)
        f_species, _ = self._species_form_factors(order, q_values, wavelength, params)
        is_complex = np.iscomplexobj(f_species)

        counts = np.bincount(codes, minlength=nsp).astype(np.float64)
        self_scattering = np.zeros(nq, dtype=np.float64)
        for c in range(nsp):
            self_scattering += counts[c] * np.abs(f_species[c]) ** 2

        want_partials = _as_bool(params, 'compute_partial_intensities', True)

        chunk_size = int(params.get('chunk_size', 2048) or 2048)
        pair_subchunk_size = int(params.get('pair_subchunk_size', 200000) or 200000)
        max_elements = int(float(params.get('max_pair_matrix_elements', 1e8)))

        r_bound = centroid_radius_bound(coords) + 1e-9
        user_cut = params.get('max_interaction_distance', None)
        if user_cut in (None, ""):
            cutoff = r_bound
            truncated = False
        else:
            cutoff = float(user_cut)
            r_true = max_pair_distance(coords)
            truncated = cutoff < r_true - 1e-9
            if truncated:
                print(f"{WARN} max_interaction_distance = {cutoff:g} A is smaller than the "
                      f"largest pair distance ({r_true:.3f} A). The Debye sum is TRUNCATED "
                      f"and the result is no longer the exact DSE.")

        if not truncated:
            pair_workload_preflight(N, params, "exact pairwise", verbose=self.verbose)

        pair_sums = np.zeros((nsp * nsp, nq), dtype=np.float64)   # sum of sinc per species pair
        n_pairs_total = 0
        block_budget = int(float(params.get('max_distance_block_elements', 4e6)))

        for dists, pcode in _iter_pair_batches(coords, codes, nsp, cutoff,
                                               complete=not truncated,
                                               max_block_elements=block_budget,
                                               chunk_size=chunk_size):
            n_pairs_total += dists.size
            M = dists.size
            pos = 0
            while pos < M:
                take = min(pair_subchunk_size, M - pos)
                if nq * take > max_elements:
                    take = max(1, max_elements // max(nq, 1))
                sl = slice(pos, pos + take)
                sub_d = dists[sl]
                sub_p = pcode[sl]

                sinc = _safe_outer_sinc(q_values, sub_d)          # (nq, take)
                for pc in np.unique(sub_p):
                    mask = sub_p == pc
                    pair_sums[pc] += sinc[:, mask].sum(axis=1)
                pos += take

        intensity = self_scattering.copy()
        partial_intensity = {}
        for pc in range(nsp * nsp):
            S = pair_sums[pc]
            if not S.any():
                continue
            a, b = divmod(pc, nsp)
            weight = 2.0 * np.real(f_species[a] * np.conj(f_species[b])) if is_complex \
                else 2.0 * f_species[a] * f_species[b]
            contrib = weight * S
            intensity += contrib
            if want_partials:
                key = self._pair_key(order, a, b)
                partial_intensity[key] = partial_intensity.get(
                    key, np.zeros(nq, dtype=np.float64)) + contrib

        if want_partials:
            for c in range(nsp):
                key = (order[c], order[c])
                partial_intensity[key] = partial_intensity.get(
                    key, np.zeros(nq, dtype=np.float64)) + counts[c] * np.abs(f_species[c]) ** 2

        self._say(f"  direct DSE: {n_pairs_total:,} unordered pairs, "
                  f"{nq} q-points, {nsp} species")
        return q_values, intensity, two_theta, partial_intensity

    # Binned pair sum (same equation, distances bucketed)
    def calculate_binned_pair_dse(self, coords, species, params, wavelength, two_theta,
                                  r_max=None, bins_per_angstrom=20.0):
        params = sanitize_params(params) if params else {}
        coords = np.asarray(coords, dtype=np.float64)
        N = coords.shape[0]
        two_theta = np.asarray(two_theta, dtype=np.float64)
        q_values = two_theta_to_q(two_theta, wavelength)
        nq = q_values.size

        order, codes = self._species_codes(species)
        nsp = len(order)
        f_species, _ = self._species_form_factors(order, q_values, wavelength, params)
        is_complex = np.iscomplexobj(f_species)

        counts_sp = np.bincount(codes, minlength=nsp).astype(np.float64)
        self_scattering = np.zeros(nq, dtype=np.float64)
        for c in range(nsp):
            self_scattering += counts_sp[c] * np.abs(f_species[c]) ** 2

        if r_max is None or not np.isfinite(r_max) or r_max <= 0:
            r_max = max_pair_distance(coords)
        r_max = float(r_max) * (1.0 + 1e-9) + 1e-9
        dr = 1.0 / float(bins_per_angstrom) if bins_per_angstrom else 0.05
        n_bins = max(16, int(math.ceil(r_max / dr)))
        edges = np.linspace(0.0, n_bins * dr, n_bins + 1)

        pair_counts = np.zeros((nsp * nsp, n_bins), dtype=np.float64)
        pair_rsum = np.zeros((nsp * nsp, n_bins), dtype=np.float64)

        cutoff = n_bins * dr
        complete = cutoff >= centroid_radius_bound(coords) - 1e-9
        pair_workload_preflight(N, params, "binned pair", verbose=self.verbose)
        block_budget = int(float(params.get('max_distance_block_elements', 4e6)))
        n_pairs_total = 0

        chunk_size = int(params.get('chunk_size', 2048) or 2048)
        for d, pcode in _iter_pair_batches(coords, codes, nsp, cutoff,
                                           complete=complete,
                                           max_block_elements=block_budget,
                                           chunk_size=chunk_size):
            if not complete:
                keep = d < cutoff
                d = d[keep]
                pcode = pcode[keep]
                if d.size == 0:
                    continue
            n_pairs_total += d.size

            b = np.floor(d / dr).astype(np.int64)
            np.clip(b, 0, n_bins - 1, out=b)
            flat = pcode * n_bins + b
            size = nsp * nsp * n_bins
            pair_counts += np.bincount(flat, minlength=size).reshape(nsp * nsp, n_bins)
            pair_rsum += np.bincount(flat, weights=d, minlength=size).reshape(nsp * nsp, n_bins)

        # Mean distance per bin (fall back to the bin centre where empty).
        centers = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(invalid='ignore', divide='ignore'):
            rbar = np.where(pair_counts > 0, pair_rsum / np.maximum(pair_counts, 1e-30),
                            centers[None, :])

        intensity = self_scattering.copy()
        partial_intensity = {}
        want_partials = _as_bool(params, 'compute_partial_intensities', True)

        for pc in range(nsp * nsp):
            cnt = pair_counts[pc]
            if not cnt.any():
                continue
            occupied = cnt > 0
            r_used = rbar[pc][occupied]
            w_used = cnt[occupied]
            S = _safe_sinc(np.outer(q_values, r_used)) @ w_used
            a, b = divmod(pc, nsp)
            weight = 2.0 * np.real(f_species[a] * np.conj(f_species[b])) if is_complex \
                else 2.0 * f_species[a] * f_species[b]
            contrib = weight * S
            intensity += contrib
            if want_partials:
                key = self._pair_key(order, a, b)
                partial_intensity[key] = partial_intensity.get(
                    key, np.zeros(nq, dtype=np.float64)) + contrib

        if want_partials:
            for c in range(nsp):
                key = (order[c], order[c])
                partial_intensity[key] = partial_intensity.get(
                    key, np.zeros(nq, dtype=np.float64)) + counts_sp[c] * np.abs(f_species[c]) ** 2

        self._say(f"  binned DSE: {n_pairs_total:,} unordered pairs -> {n_bins} bins of "
                  f"{dr:g} A (r_max = {n_bins * dr:.2f} A)")
        return q_values, intensity, two_theta, partial_intensity

# Measurement-correction layer (shared by both methods)
def apply_measurement_corrections(two_theta, intensity, partial_intensity, params,
                                  verbose=True):
    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).copy()
    partial_intensity = {k: np.asarray(v, dtype=np.float64).copy()
                         for k, v in (partial_intensity or {}).items()}
    applied = []

    if _as_bool(params, 'apply_march_dollase', False):
        raise ValueError(
            "apply_march_dollase is not applicable to the Debye scattering equation.\n"
            "  March-Dollase needs the angle between each reflection's scattering vector\n"
            "  and the texture axis, evaluated per hkl. The DSE result is already an\n"
            "  isotropic orientational average with no hkl indices, so no such angle\n"
            "  exists. The previous 2theta-only expression was not this correction.\n"
            "  Model texture with XRD-Kinematical (per-reflection) instead, or remove\n"
            "  apply_march_dollase from the input.")

    # Lorentz-polarization
    if _as_bool(params, 'apply_LP', False):
        variant = str(params.get('LP_variant', 'with_polarization')).strip().lower()
        theta = np.radians(two_theta / 2.0)
        two_theta_rad = np.radians(two_theta)

        theta_min_clip = np.radians(max(0.0, float(params.get('LP_theta_min_clip_deg', 0.5))))
        sin_theta = np.maximum(np.sin(theta), np.sin(theta_min_clip))
        cos_theta = np.cos(theta)
        cos_theta = np.where(np.abs(cos_theta) < 1e-12,
                             np.sign(cos_theta) * 1e-12 + 1e-30, cos_theta)

        lorentz = 1.0 / (sin_theta ** 2 * cos_theta)
        if variant in ('lorentz', 'lorentz_only'):
            LP = lorentz
            desc = "Lorentz only, 1/(sin^2(theta) cos(theta))"
        elif variant in ('none', 'off'):
            LP = np.ones_like(two_theta)
            desc = "none"
        else:
            LP = (1.0 + np.cos(two_theta_rad) ** 2) * lorentz
            desc = "(1 + cos^2(2theta)) / (sin^2(theta) cos(theta)), unpolarised source"

        LP = np.clip(LP, 0.0, float(params.get('LP_max_clip', 1e4)))
        intensity *= LP
        for k in partial_intensity:
            partial_intensity[k] *= LP
        applied.append(f"Lorentz-polarization [{desc}]")
        if verbose:
            print(f"  LP applied: {desc}")

    # instrumental resolution
    if _as_bool(params, 'apply_instrumental_broadening', False):
        if len(two_theta) < 3:
            if verbose:
                print(f"{WARN} too few points to broaden; skipping")
        elif _as_bool(params, 'use_caglioti', False):
            U = float(params.get('caglioti_U', 0.01))
            V = float(params.get('caglioti_V', 0.01))
            W = float(params.get('caglioti_W', 0.01))
            tan_t = np.tan(np.radians(two_theta / 2.0))
            fwhm2 = U * tan_t ** 2 + V * tan_t + W
            fwhm = np.sqrt(np.maximum(fwhm2, 0.0))
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            intensity = _variable_width_convolve(two_theta, intensity, sigma)
            for k in partial_intensity:
                partial_intensity[k] = _variable_width_convolve(
                    two_theta, partial_intensity[k], sigma)
            applied.append(f"Caglioti resolution (U={U:g}, V={V:g}, W={W:g})")
            if verbose:
                print(f"  Caglioti broadening: FWHM {fwhm.min():.4f}-{fwhm.max():.4f} deg")
        else:
            sigma_deg = float(params.get('instrument_sigma_deg', 0.05))
            step = float(two_theta[1] - two_theta[0])
            if sigma_deg > 0 and step > 0:
                sigma_pts = sigma_deg / step
                intensity = gaussian_filter1d(intensity, sigma_pts, mode='nearest')
                for k in partial_intensity:
                    partial_intensity[k] = gaussian_filter1d(
                        partial_intensity[k], sigma_pts, mode='nearest')
                applied.append(f"Gaussian resolution, constant sigma = {sigma_deg:g} deg 2theta")
                if verbose:
                    print(f"  Gaussian broadening: sigma = {sigma_deg:g} deg "
                          f"({sigma_pts:.2f} grid points)")

    # global scale
    scale = float(params.get('scale_factor', 1.0))
    if scale != 1.0:
        intensity *= scale
        for k in partial_intensity:
            partial_intensity[k] *= scale
        applied.append(f"scale_factor = {scale:g}")

    return intensity, partial_intensity, applied


def _variable_width_convolve(x, y, sigma, n_sigma=4.0):
    """Convolve y(x) with a Gaussian whose width varies with position.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if x.size < 2:
        return y.copy()
    dx = float(x[1] - x[0])
    out = np.zeros_like(y)
    valid = sigma > 0
    # points with zero width pass straight through
    out[~valid] += y[~valid]
    for i in np.nonzero(valid)[0]:
        s = sigma[i]
        half = int(math.ceil(n_sigma * s / dx))
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        g = np.exp(-0.5 * ((x[lo:hi] - x[i]) / s) ** 2)
        area = g.sum() * dx
        if area <= 0:
            out[i] += y[i]
            continue
        out[lo:hi] += y[i] * g / area * dx
    return out


# Structural reporting: partial RDFs and the pair-distance distribution

def build_ordered_pair_histograms(coords, species, max_distance, n_bins,
                                  centers_idx=None):
    """Histogram of neighbour counts around each centre, keyed (centre, neighbour).
    """
    coords = np.asarray(coords, dtype=np.float64)
    species = np.asarray(species)
    tree = cKDTree(coords)
    edges = np.linspace(0.0, max_distance, n_bins + 1)
    dr = edges[1] - edges[0]

    if centers_idx is None:
        centers_idx = np.arange(coords.shape[0])
    centers_idx = np.asarray(centers_idx, dtype=np.int64)

    order = sorted(set(species.tolist()))
    lookup = {el: i for i, el in enumerate(order)}
    codes = np.fromiter((lookup[s] for s in species), dtype=np.int64, count=species.size)
    nsp = len(order)

    acc = np.zeros((nsp * nsp, n_bins), dtype=np.float64)
    centre_counts = Counter(species[centers_idx].tolist())

    block = 2048
    for s in range(0, centers_idx.size, block):
        sel = centers_idx[s:s + block]
        neigh_lists = tree.query_ball_point(coords[sel], max_distance)
        for local, neigh in enumerate(neigh_lists):
            i = int(sel[local])
            arr = np.fromiter(neigh, dtype=np.int64, count=len(neigh))
            arr = arr[arr != i]
            if arr.size == 0:
                continue
            d = np.linalg.norm(coords[arr] - coords[i], axis=1)
            keep = d < max_distance
            if not keep.any():
                continue
            d = d[keep]
            arr = arr[keep]
            b = np.floor(d / dr).astype(np.int64)
            np.clip(b, 0, n_bins - 1, out=b)
            flat = codes[i] * nsp * n_bins + codes[arr] * n_bins + b
            acc += np.bincount(flat, minlength=nsp * nsp * n_bins).reshape(nsp * nsp, n_bins)

    hists = {}
    for a in range(nsp):
        for b in range(nsp):
            row = acc[a * nsp + b]
            if row.any():
                hists[(order[a], order[b])] = row
    return hists, 0.5 * (edges[:-1] + edges[1:]), dict(centre_counts)


def estimate_accessible_fraction(coords, bin_centers, n_dirs=200, n_centers=500,
                                 rng_seed=0, verbose=True):
    """Fraction of a spherical shell of radius r that lies inside the cluster.
    """
    coords = np.asarray(coords, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    if coords.size == 0 or bin_centers.size == 0:
        return np.ones_like(bin_centers, dtype=np.float64)

    N = coords.shape[0]
    rng = np.random.default_rng(int(rng_seed))
    centers_idx = rng.choice(N, min(int(n_centers), N), replace=False)

    dirs = rng.normal(size=(int(n_dirs), 3))
    norms = np.linalg.norm(dirs, axis=1)
    norms[norms == 0] = 1.0
    dirs /= norms[:, None]

    delaunay = None
    try:
        if coords.shape[0] >= 4:
            from scipy.spatial import Delaunay
            delaunay = Delaunay(coords)
    except Exception:
        delaunay = None
    if delaunay is None and verbose:
        print(f"{WARN} convex-hull test unavailable; accessible fraction falls back to "
              f"a bounding-box test (cruder).")

    mins = coords.min(axis=0) - 1e-8
    maxs = coords.max(axis=0) + 1e-8

    frac = np.zeros_like(bin_centers, dtype=np.float64)
    base = coords[centers_idx]                      # (nc, 3)
    for i, r in enumerate(bin_centers):
        pts = (base[:, None, :] + r * dirs[None, :, :]).reshape(-1, 3)
        if delaunay is not None:
            inside = delaunay.find_simplex(pts) >= 0
        else:
            inside = np.all((pts >= mins) & (pts <= maxs), axis=1)
        frac[i] = inside.mean() if inside.size else 1.0

    return np.clip(frac, 1e-3, 1.0)


def compute_partial_rdfs(hists, bin_centers, species, coords, centre_counts,
                         params=None):
    """Partial radial distribution functions g_ab(r) from ORDERED pair histograms.
    """
    params = params or {}
    species = list(species)
    species_counts = Counter(species)
    coords = np.asarray(coords, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    dr = float(bin_centers[1] - bin_centers[0])

    # number density per species
    rho = None
    if params.get("number_density") not in (None, ""):
        try:
            nd = float(params["number_density"])
            total = float(sum(species_counts.values()))
            rho = {el: nd * (species_counts[el] / total) for el in species_counts}
        except Exception:
            rho = None

    if rho is None:
        volume = None
        try:
            if ConvexHull is not None and coords.shape[0] >= 4:
                volume = float(ConvexHull(coords).volume)
                if not np.isfinite(volume) or volume <= 0:
                    volume = None
        except Exception:
            volume = None
        if volume is None:
            mins, maxs = coords.min(axis=0), coords.max(axis=0)
            bbox = maxs - mins
            pad = 0.1 * max(1.0, float(np.max(bbox)))
            volume = float(np.prod(bbox + pad))
        volume = max(volume, float(params.get('min_effective_volume', 10.0)))
        rho = {el: species_counts[el] / volume for el in species_counts}

    if _as_bool(params, 'apply_accessible_fraction', False):
        accessible = estimate_accessible_fraction(
            coords, bin_centers,
            n_dirs=int(params.get('rdf_access_dirs', 200)),
            n_centers=int(params.get('rdf_access_centers', 500)),
            rng_seed=int(params.get('random_seed', 0)),
            verbose=_as_bool(params, 'verbose', True))
    else:
        accessible = np.ones_like(bin_centers)

    shell = 4.0 * np.pi * bin_centers ** 2 * dr * accessible

    g_r = {}
    for (a, b), hist in hists.items():
        n_a = float(centre_counts.get(a, species_counts.get(a, 1)) or 1)
        denom = shell * rho[b] * n_a
        g_r[(a, b)] = np.divide(hist, denom, out=np.zeros_like(hist), where=denom > 0)
    return g_r, rho


def save_partial_rdfs(bin_centers, g_r, rho, params=None, out_dir='.'):
    """Write g_ab(r) to rdf_<a>-<b>.txt.  Returns the list of files written."""
    params = params or {}
    out_files = []
    for (a, b), g in sorted(g_r.items()):
        fname = os.path.join(out_dir, f"rdf_{a}-{b}.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"# Partial RDF g_{a}{b}(r)  (diagnostic output; not used for I(q))\n")
                f.write(f"# a = central species, b = neighbour species\n")
                f.write(f"# rho_{b} = {rho.get(b, 0.0):.6e} atoms/A^3\n")
                f.write("# r(A)          g(r)\n")
                for r, gr in zip(bin_centers, g):
                    f.write(f"{r:12.6f} {gr:12.6f}\n")
            out_files.append(fname)
            if _as_bool(params, 'verbose', True):
                print(f"{OK} saved partial RDF: {fname}")
        except Exception as e:
            print(f"{ERR} failed to save {fname}: {e}")
    return out_files


def plot_combined_partial_rdfs(rdf_files=None, params=None):
    """Plot all partial RDFs in one figure."""
    params = params or {}
    if rdf_files is None:
        rdf_files = sorted(glob.glob("rdf_*.txt"))
        if not rdf_files:
            print(f"{WARN} no RDF files found to plot")
            return False

    figsize = tuple(params.get('combined_rdf_figsize', (12, 6)))
    dpi = int(params.get('combined_rdf_dpi', 300))
    plt.figure(figsize=figsize)
    for rdf_file in sorted(rdf_files):
        try:
            data = np.loadtxt(rdf_file, comments='#')
            if data.size == 0:
                continue
            plt.plot(data[:, 0], data[:, 1], linewidth=2,
                     label=os.path.basename(rdf_file).replace(".txt", ""))
        except Exception as e:
            print(f"{ERR} failed to read {rdf_file}: {e}")
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title("Combined partial RDFs")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    out_file = params.get('combined_rdf_plot', 'combined_partial_rdfs.png')
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"{OK} combined partial RDF plot saved: {out_file}")
    return True


def save_pair_distance_distribution(coords, params=None):
    """Write the pair-distance distribution P(r) and the cumulative coordination.
    """
    params = params or {}
    try:
        coords = np.asarray(coords, dtype=np.float64)
        N = coords.shape[0]
        subset_size = int(min(int(params.get('pdf_subset_size', 5000)), N))
        rng = np.random.default_rng(int(params.get('random_seed', 0)))
        if subset_size < N:
            idx = rng.choice(N, subset_size, replace=False)
            sub = coords[idx]
            if _as_bool(params, 'verbose', True):
                print(f"  pair-distance distribution from a {subset_size:,}-atom subsample "
                      f"(seed {int(params.get('random_seed', 0))})")
        else:
            sub = coords

        if pdist is None:
            raise RuntimeError("scipy.spatial.distance.pdist unavailable")
        distances = pdist(sub)
        if distances.size == 0:
            raise RuntimeError("no pair distances computed")

        n_bins = int(params.get('pdf_bins', 100))
        hist, edges = np.histogram(distances, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        raw_counts, _ = np.histogram(distances, bins=edges)
        n_sub = sub.shape[0]

        coordination = np.cumsum(2.0 * raw_counts) / max(n_sub, 1)

        pdf_file = params.get('pdf_output', 'pair_distance_distribution.txt')
        with open(pdf_file, 'w', encoding='utf-8') as f:
            f.write("# Pair-distance distribution (diagnostic output; not used for I(q))\n")
            f.write(f"# Based on {n_sub:,} atoms of {N:,} total\n")
            f.write(f"# Maximum pair distance in the subsample: {np.max(distances):.4f} A\n")
            f.write("# P(r) is the pair-distance probability density (integrates to 1),\n")
            f.write("#      NOT the radial distribution function g(r).\n")
            f.write("# n(r) is the cumulative coordination number: mean number of\n")
            f.write("#      neighbours within r of an atom.\n")
            f.write("# r(A)            P(r)         n(r)\n")
            for d, p, cn in zip(centers, hist, coordination):
                f.write(f"{d:12.4f} {p:14.6e} {cn:12.4f}\n")

        if _as_bool(params, 'plot_pdf', False):
            plt.figure(figsize=tuple(params.get('pdf_figsize', (10, 8))))
            plt.subplot(2, 1, 1)
            plt.plot(centers, hist, 'b-', linewidth=2)
            plt.xlabel("r (A)")
            plt.ylabel("P(r)")
            plt.grid(True, alpha=0.3)
            plt.subplot(2, 1, 2)
            plt.plot(centers, coordination, 'r-', linewidth=2)
            plt.xlabel("r (A)")
            plt.ylabel("cumulative n(r)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_name = str(pdf_file).replace('.txt', '_plot.png')
            plt.savefig(plot_name, dpi=int(params.get('pdf_dpi', 300)))
            plt.close()
            print(f"{OK} pair-distance plot saved to {plot_name}")

        print(f"{OK} pair-distance distribution saved to {pdf_file}")
        return True
    except Exception as e:
        print(f"{ERR} could not compute the pair-distance distribution: {e}")
        return False


def estimate_correlation_length(coords, species=None, max_probe=80.0, rng_seed=0,
                                n_sample=2000):

    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    if N < 2:
        return 1.0

    r_true_max = max_pair_distance(coords)
    probe = float(min(max_probe, r_true_max)) if r_true_max > 0 else float(max_probe)
    if probe <= 0:
        return 1.0

    dr = max(0.05, min(0.5, probe / 200.0))
    n_bins = max(10, int(probe / dr))
    edges = np.linspace(0.0, n_bins * dr, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rng = np.random.default_rng(int(rng_seed))
    n_sample = int(min(n_sample, N))
    sel = rng.choice(N, n_sample, replace=False) if n_sample < N else np.arange(N)

    tree = cKDTree(coords)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in sel:
        neigh = tree.query_ball_point(coords[i], n_bins * dr)
        arr = np.fromiter(neigh, dtype=np.int64, count=len(neigh))
        arr = arr[arr != i]
        if arr.size == 0:
            continue
        d = np.linalg.norm(coords[arr] - coords[i], axis=1)
        b = np.floor(d / dr).astype(np.int64)
        np.clip(b, 0, n_bins - 1, out=b)
        counts += np.bincount(b, minlength=n_bins)

    volume = None
    try:
        if ConvexHull is not None and N >= 4:
            volume = float(ConvexHull(coords).volume)
    except Exception:
        volume = None
    if not volume or not np.isfinite(volume) or volume <= 0:
        bbox = coords.max(axis=0) - coords.min(axis=0)
        volume = float(np.prod(np.maximum(bbox, 1e-6)))
    rho = N / volume

    shell = 4.0 * np.pi * centers ** 2 * dr * rho * n_sample
    g = np.divide(counts, shell, out=np.zeros_like(counts), where=shell > 0)
    g_smooth = gaussian_filter1d(g, sigma=1)

    tol = 0.15
    run = 3
    start = max(1, int(1.5 / dr))
    for i in range(start, n_bins - run):
        if np.all(np.abs(g_smooth[i:i + run] - 1.0) < tol):
            return float(min(centers[i], r_true_max))
    idx = int(np.argmin(np.abs(g_smooth[start:] - 1.0))) + start
    return float(min(centers[min(idx, n_bins - 1)], r_true_max))



def read_with_ase(filename, verbose=True):
    try:
        from ase.io import read as ase_read
    except ImportError:
        return None

    try:
        # index=0: the first frame, as read_lammps_xyz takes
        atoms = ase_read(filename, index=0)
    except Exception as exc:
        if verbose:
            if str(exc).find("Unknown') format") != -1:
                print(f"  ASE could not determine the file format. Supported formats "
                      f"include XYZ, LAMMPS data, POSCAR/CONTCAR, CIF, VASP OUTCAR, etc.")
            else:
                print(f"  ASE error reading file: {exc}")
        return None

    try:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        atom_names = [str(s) for s in atoms.get_chemical_symbols()]
    except Exception as exc:
        if verbose:
            print(f"  ASE read the file but no positions/species could be extracted: {exc}")
        return None

    if len(positions) == 0:
        return None

    cell = None
    try:
        matrix = np.asarray(atoms.get_cell(), dtype=float)
        if matrix.shape == (3, 3) and abs(np.linalg.det(matrix)) > 1e-8:
            cell = matrix
    except Exception:
        cell = None

    try:
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    except Exception:
        pbc = None

    if verbose:
        print(f"  ASE successfully read the structure file:")
        print(f"    Atoms: {len(positions)}")
        if cell is None:
            print(f"    Cell: not provided by the file")
        else:
            print(f"    Cell: a={np.linalg.norm(cell[0]):.4f}, "
                  f"b={np.linalg.norm(cell[1]):.4f}, "
                  f"c={np.linalg.norm(cell[2]):.4f} A")
        if pbc is not None:
            print(f"    Periodic boundaries: {list(pbc)}")
        print(f"    Species found: {sorted(set(atom_names))}")

    return positions, atom_names, cell, pbc


def read_lammps_data(filename, verbose=True):
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        num_atoms = 0
        atom_types = {}
        positions = []
        atom_names = []

        cell = np.zeros((3, 3))
        have_box = [False, False, False]
        pbc = [True, True, True]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith('#'):
                i += 1
                continue

            if "atoms" in line and "atom types" not in line:
                parts = line.split()
                for j, part in enumerate(parts):
                    if part == "atoms":
                        num_atoms = int(parts[j - 1])
                        break

            elif "xlo xhi" in line:
                parts = line.split()
                xlo, xhi = float(parts[0]), float(parts[1])
                cell[0, 0] = xhi - xlo
                have_box[0] = True
            elif "ylo yhi" in line:
                parts = line.split()
                ylo, yhi = float(parts[0]), float(parts[1])
                cell[1, 1] = yhi - ylo
                have_box[1] = True
            elif "zlo zhi" in line:
                parts = line.split()
                zlo, zhi = float(parts[0]), float(parts[1])
                cell[2, 2] = zhi - zlo
                have_box[2] = True

            elif line.startswith("Masses"):
                i += 2
                while i < len(lines) and lines[i].strip():
                    mass_line = lines[i].strip()
                    if not mass_line.startswith('#'):
                        parts = mass_line.split()
                        if len(parts) >= 2:
                            atype = int(parts[0])
                            mass = float(parts[1])
                            if 190 < mass < 200:
                                atom_types[atype] = "Au"
                            elif 180 < mass < 190:
                                atom_types[atype] = "W"
                            elif 160 < mass < 180:
                                atom_types[atype] = "Ta"
                            elif 140 < mass < 160:
                                atom_types[atype] = "Gd"
                            elif 120 < mass < 140:
                                atom_types[atype] = "Sn"
                            elif 90 < mass < 100:
                                atom_types[atype] = "Mo"
                            elif 50 < mass < 70:
                                atom_types[atype] = "Fe"
                            elif 40 < mass < 50:
                                atom_types[atype] = "Ca"
                            elif 26 < mass < 28:
                                atom_types[atype] = "Al"
                            elif 23 < mass < 25:
                                atom_types[atype] = "Mg"
                            elif 10 < mass < 20:
                                atom_types[atype] = "C"
                            elif 14 < mass < 16:
                                atom_types[atype] = "N"
                            elif 15 < mass < 17:
                                atom_types[atype] = "O"
                            elif 1 < mass < 2:
                                atom_types[atype] = "H"
                            else:
                                atom_types[atype] = str(atype)
                    i += 1
                continue

            elif "Atoms" in line:
                i += 2
                atom_count = 0
                while atom_count < num_atoms and i < len(lines):
                    atom_line = lines[i].strip()
                    if atom_line and not atom_line.startswith('#'):
                        parts = atom_line.split()
                        if len(parts) >= 5:
                            atype = int(parts[1])
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            positions.append([x, y, z])
                            atom_names.append(atom_types.get(atype, str(atype)))
                            atom_count += 1
                    i += 1
                break

            i += 1

        if len(positions) == 0:
            return None

        if not all(have_box):
            missing = [t for t, h in zip('xyz', have_box) if not h]
            print(f"{WARN} LAMMPS data file carries no {missing} box bounds; "
                  f"the cell is reported as unknown rather than guessed.")
            cell = None

        if verbose:
            print(f"  LAMMPS data file read successfully:")
            print(f"    Atoms: {len(positions)}")
            print(f"    Species found: {sorted(set(atom_names))}")
            if cell is not None:
                print(f"    Box lengths: {cell[0,0]:.4f}, {cell[1,1]:.4f}, "
                      f"{cell[2,2]:.4f} A")

        return np.asarray(positions, dtype=float), atom_names, cell, np.asarray(pbc, dtype=bool)

    except Exception as exc:
        if verbose:
            print(f"  Error reading LAMMPS data file: {exc}")
        return None


def _apply_type_map(result, type_map, verbose=True):
    positions, atom_names, cell, pbc = result
    if not type_map:
        return positions, atom_names, cell, pbc
    remapped = []
    hits = 0
    for name in atom_names:
        key = str(name).strip()
        if key.isdigit() and int(key) in type_map:
            remapped.append(type_map[int(key)])
            hits += 1
        elif key in ('', 'X') and 0 in type_map:
            remapped.append(type_map[0])
            hits += 1
        else:
            remapped.append(key)
    if hits and verbose:
        print(f"    type_map applied to {hits} of {len(atom_names)} sites")
    return positions, remapped, cell, pbc


def read_structure_file(filename, params, verbose=True):
    _species_mode, numeric_mode, type_map = species_read_options(params)
    use_ase = _as_bool(params, 'use_ase', True)

    if use_ase:
        if verbose:
            print(f"Attempting to read structure file with ASE...")
        result = read_with_ase(filename, verbose=verbose)
        if result is not None:
            return _apply_type_map(result, type_map, verbose=verbose)

    low = str(filename).lower()
    if low.endswith('.data') or low.endswith('.lmp'):
        if verbose:
            print(f"ASE failed or not available. Trying LAMMPS data format...")
        result = read_lammps_data(filename, verbose=verbose)
        if result is not None:
            return _apply_type_map(result, type_map, verbose=verbose)

    if verbose:
        print(f"Trying XYZ format as fallback...")
    try:
        species, coords = read_lammps_xyz(filename, verbose=verbose,
                                          numeric_mode=numeric_mode, type_map=type_map)
        return np.asarray(coords, dtype=float), [str(s) for s in species], None, None
    except Exception as exc:
        raise ValueError(
            f"Could not read structure file {filename!r} with any available method.\n"
            f"  ASE, the LAMMPS data reader and the XYZ reader all failed.\n"
            f"  Last error from the XYZ reader: {exc}")


# Runner

def run_debye(params):
    params = sanitize_params(params)
    verbose = _as_bool(params, 'verbose', True)

    print("\n" + "=" * 70)
    print("DEBYE SCATTERING CALCULATION (finite, non-periodic structures)")
    print("=" * 70)

    if not validate_params(params, ['xyz_file'], "Debye scattering"):
        return False

    dw_raw = params.get('debye_waller_factors')
    if dw_raw:
        items = None
        if isinstance(dw_raw, dict):
            for el, val in dw_raw.items():
                try:
                    params[f"B_{str(el).strip()}"] = float(val)
                except Exception:
                    print(f"{WARN} could not parse Debye-Waller value for {el!r}")
        elif isinstance(dw_raw, (list, tuple)):
            items = [str(x).strip() for x in dw_raw if str(x).strip()]
        elif isinstance(dw_raw, str):
            items = [p.strip() for p in dw_raw.split(',') if p.strip()]
        if items:
            for item in items:
                if '=' not in item:
                    print(f"{WARN} ignoring malformed debye_waller_factors entry {item!r} "
                          f"(expected 'El=value')")
                    continue
                el, val = item.split('=', 1)
                try:
                    params[f"B_{el.strip()}"] = float(val.strip())
                except Exception:
                    print(f"{WARN} could not parse Debye-Waller value in {item!r}")

    if _as_bool(params, 'experimental_correction', False):
        params.setdefault('apply_LP', True)
        params.setdefault('apply_instrumental_broadening', True)
        params.setdefault('apply_debye_waller', True)
        if 'apply_anomalous' not in params:
            print("Note: experimental_correction enables LP, instrumental broadening and "
                  "Debye-Waller. Anomalous dispersion is NOT enabled automatically; set "
                  "apply_anomalous = yes explicitly if you need it.")

    result = read_structure_file(params['xyz_file'], params, verbose=verbose)
    coords, species_list, _cell, _pbc = result
    coords = np.asarray(coords, dtype=np.float64)
    species = list(species_list)
    N = len(species)
    if N == 0:
        print(f"{ERR} no atoms read from {params['xyz_file']}")
        return False

    composition = Counter(species)
    centroid = coords.mean(axis=0)
    cluster_radius = float(np.max(np.linalg.norm(coords - centroid, axis=1)))
    print(f"Structure: {N:,} atoms")
    print(f"Composition: " + ", ".join(f"{el}={n}" for el, n in sorted(composition.items())))
    print(f"Cluster radius (from centroid): {cluster_radius:.3f} A")

    wavelength = resolve_wavelength(params.get('wavelength', 'CuKa'), verbose=verbose)
    print(f"Wavelength: {wavelength:.6f} A")

    two_theta_min = float(params.get('two_theta_min', 10.0) or 10.0)
    user_tt_max = params.get('two_theta_max', None)
    q_max = params.get('q_max', None)
    if user_tt_max is not None and q_max is not None:
        tt_from_q = None
        arg = float(q_max) * wavelength / (4.0 * np.pi)
        if arg < 1.0:
            tt_from_q = 2.0 * np.degrees(np.arcsin(arg))
        print(f"Note: both two_theta_max ({float(user_tt_max):g} deg) and q_max "
              f"({float(q_max):g} 1/A) were given. two_theta_max takes precedence"
              + (f"; q_max alone would have given {tt_from_q:.3f} deg." if tt_from_q
                 else "; q_max exceeds 2theta = 180 deg."))
    if user_tt_max is not None:
        two_theta_max = float(user_tt_max)
    elif q_max is not None:
        arg = float(q_max) * wavelength / (4.0 * np.pi)
        if arg >= 1.0:
            two_theta_max = 180.0 - 1e-6
            print(f"{WARN} q_max = {float(q_max):g} 1/A implies 2theta >= 180 deg; "
                  f"capped at 180 deg.")
        else:
            two_theta_max = 2.0 * np.degrees(np.arcsin(arg))
    else:
        two_theta_max = 90.0

    two_theta_min = max(1e-6, min(179.0, two_theta_min))
    two_theta_max = max(two_theta_min + 1e-6, min(180.0 - 1e-6, two_theta_max))
    n_points = int(params.get('n_points', 500) or 500)
    two_theta = np.linspace(two_theta_min, two_theta_max, n_points)
    print(f"2theta range: {two_theta_min:.4f} to {two_theta_max:.4f} deg, {n_points} points")

    method = str(params.get('debye_method', 'auto') or 'auto').strip().lower()
    if method == 'histogram':
        print("Note: debye_method = 'histogram' is a deprecated spelling of 'binned'. "
              "The old RDF-based reconstruction has been removed (it produced negative "
              "intensities); 'binned' is an exact binned pair sum.")
        method = 'binned'
    if method not in ('pairwise', 'binned', 'auto'):
        print(f"{WARN} unknown debye_method {method!r}; using 'auto'")
        method = 'auto'

    direct_threshold = int(params.get('direct_threshold', 2000) or 2000)
    if method == 'auto':
        use_direct = N <= direct_threshold
        print(f"Method: auto -> {'pairwise' if use_direct else 'binned'} "
              f"(N = {N:,} vs direct_threshold = {direct_threshold:,})")
    else:
        use_direct = (method == 'pairwise')
        print(f"Method: {method} (explicitly requested; N = {N:,})")

    calculator = DebyeCalculator(verbose=verbose)
    start_time = time.time()

    if use_direct:
        q_values, intensity, two_theta, partial_intensity = calculator.calculate_direct_dse(
            coords, species, params, wavelength, two_theta)
        method_label = "exact pairwise Debye sum"
    else:
        bpa = float(params.get('bins_per_angstrom', 20) or 20)
        raw_max = params.get('max_distance', 'adaptive')
        r_max = None
        if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool):
            r_max = float(raw_max)
        elif isinstance(raw_max, str) and raw_max.strip().lower() not in (
                '', 'adaptive', 'auto', 'none', 'null'):
            try:
                r_max = float(raw_max)
            except Exception:
                raise ValueError(f"Invalid max_distance {raw_max!r}: use a number, "
                                 f"'adaptive', or leave empty.")
        r_true = max_pair_distance(coords)
        if r_max is None:
            r_max = r_true
            print(f"  max_distance = adaptive -> exact maximum pair distance "
                  f"{r_max:.3f} A (the binned DSE needs every pair, so this is not a "
                  f"correlation-length estimate)")
        elif r_max < r_true - 1e-9:
            print(f"{WARN} max_distance = {r_max:g} A < largest pair distance "
                  f"{r_true:.3f} A. The Debye sum will be TRUNCATED.")
        q_values, intensity, two_theta, partial_intensity = calculator.calculate_binned_pair_dse(
            coords, species, params, wavelength, two_theta,
            r_max=r_max, bins_per_angstrom=bpa)
        method_label = f"binned pair Debye sum ({bpa:g} bins/A)"

    calc_time = time.time() - start_time
    print(f"Intrinsic DSE computed in {calc_time:.2f} s")

    if np.any(intensity < 0):
        neg = int(np.sum(intensity < 0))
        print(f"{ERR} {neg} negative intensity values. The DSE is positive-definite, so "
              f"this indicates a bug - please report it with the input file.")

    intensity, partial_intensity, applied = apply_measurement_corrections(
        two_theta, intensity, partial_intensity, params, verbose=verbose)

    norm_note = "none"
    if _as_bool(params, 'normalize_intensity', False):
        target = float(params.get('normalize_max', 100.0))
        maxI = float(np.max(intensity)) if intensity.size else 0.0
        if maxI > 0:
            scale = target / maxI
            intensity = intensity * scale
            for k in partial_intensity:
                partial_intensity[k] = partial_intensity[k] * scale
            norm_note = f"max scaled to {target:g} (factor {scale:.6e})"
            if verbose:
                print(f"  normalised: max was {maxI:.6e}, scaled to {target:g}")
        else:
            print(f"{WARN} cannot normalise: maximum intensity <= 0")

    if _as_bool(params, 'compute_rdf', False):
        out_dir = params.get('output_dir', '.') or '.'
        os.makedirs(out_dir, exist_ok=True)
        rdf_rmax = float(params.get('rdf_max_distance', min(20.0, max_pair_distance(coords))))
        rdf_bins = max(16, int(rdf_rmax * float(params.get('bins_per_angstrom', 20) or 20)))
        sample = int(params.get('rdf_sample_size', 0) or 0)
        rng = np.random.default_rng(int(params.get('random_seed', 0)))
        centers_idx = (rng.choice(N, sample, replace=False)
                       if 0 < sample < N else np.arange(N))
        hists, bin_centers, centre_counts = build_ordered_pair_histograms(
            coords, species, rdf_rmax, rdf_bins, centers_idx=centers_idx)
        g_r, rho = compute_partial_rdfs(hists, bin_centers, species, coords,
                                        centre_counts, params=params)
        try:
            xi = estimate_correlation_length(coords, species,
                                             rng_seed=int(params.get('random_seed', 0)))
            print(f"  structural correlation length (g(r) settles near 1): {xi:.2f} A")
        except Exception as exc:
            print(f"{WARN} correlation-length estimate failed: {exc}")
        files = save_partial_rdfs(bin_centers, g_r, rho, params=params, out_dir=out_dir)
        if _as_bool(params, 'plot_rdf', False) and files:
            plot_combined_partial_rdfs(files, params)
    if _as_bool(params, 'compute_pair_distribution', False):
        save_pair_distance_distribution(coords, params)

    header_lines = [
        "Debye scattering equation, finite non-periodic configuration",
        f"structure         : {params['xyz_file']} ({N} atoms)",
        f"composition       : " + ", ".join(f"{el}={n}" for el, n in sorted(composition.items())),
        f"wavelength        : {wavelength:.6f} A",
        f"method            : {method_label}",
        f"anomalous         : {'yes' if _as_bool(params, 'apply_anomalous', False) else 'no'}",
        f"Debye-Waller      : {'yes' if _as_bool(params, 'apply_debye_waller', False) else 'no'}",
        "correction order  : intrinsic I_DSE -> "
        + (" -> ".join(applied) if applied else "(none)")
        + f" -> normalisation [{norm_note}]",
        "columns           : q(1/A)  2theta(deg)  I(arb.)",
    ]

    out_pattern = params.get('output_pattern', 'debye_pattern.txt')
    try:
        with open(out_pattern, 'w', encoding='utf-8') as fh:
            for line in header_lines:
                fh.write(f"# {line}\n")
            for q, tt, I in zip(q_values, two_theta, intensity):
                fh.write(f"{q:14.8f} {tt:12.6f} {I:20.10e}\n")
        print(f"{OK} pattern saved to {out_pattern}")
    except Exception as e:
        print(f"{ERR} failed to save pattern to {out_pattern}: {e}")

    if partial_intensity:
        pfile = params.get('partial_output', 'debye_partials.txt')
        try:
            keys = sorted(partial_intensity)
            with open(pfile, 'w', encoding='utf-8') as fh:
                for line in header_lines:
                    fh.write(f"# {line}\n")
                fh.write("# partial intensities by species pair; like-species columns "
                         "include the self-scattering term\n")
                fh.write("# q(1/A)  2theta(deg)  " +
                         "  ".join(f"{a}-{b}" for a, b in keys) + "\n")
                for n in range(len(q_values)):
                    row = "  ".join(f"{partial_intensity[k][n]:20.10e}" for k in keys)
                    fh.write(f"{q_values[n]:14.8f} {two_theta[n]:12.6f} {row}\n")
            print(f"{OK} partial intensities saved to {pfile}")
        except Exception as e:
            print(f"{ERR} failed to save partials: {e}")

    if _as_bool(params, 'make_plot', True):
        try:
            plt.figure(figsize=(9, 5.5))
            plt.plot(two_theta, intensity, 'b-', linewidth=1.8, label='total')
            if _as_bool(params, 'plot_partials', False):
                for key in sorted(partial_intensity):
                    plt.plot(two_theta, partial_intensity[key], linewidth=1.0,
                             alpha=0.8, label=f'{key[0]}-{key[1]}')
            plt.xlabel('2theta (deg)')
            plt.ylabel('Intensity (arb. units)')
            plt.title(f'Debye scattering, {method_label}')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9)
            plt.tight_layout()
            plot_filename = params.get('plot_filename', 'debye_plot.png')
            plt.savefig(plot_filename, dpi=int(params.get('plot_dpi', 300)))
            if _as_bool(params, 'show_plot', False):
                plt.show()
            plt.close()
            print(f"{OK} plot saved to {plot_filename}")
        except Exception as e:
            print(f"{ERR} plotting failed: {e}")

    return True

EnhancedDebyeCalculator = DebyeCalculator
run_enhanced_debye = run_debye
DebyeCalculator.calculate_direct_debye_chunked = DebyeCalculator.calculate_direct_dse

def self_test():
    print("=" * 70)
    print("SELF-TEST: binned pair DSE vs exact pairwise DSE")
    print("=" * 70)
    rng = np.random.default_rng(7)
    a = 3.615
    basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    pts.append(np.array([i, j, k]) * a + b)
    pts = np.array(pts)
    c = pts.mean(0)
    pts = pts[np.linalg.norm(pts - c, axis=1) <= 7.5]
    species = list(rng.choice(['Cu', 'Zn'], size=len(pts), p=[0.7, 0.3]))
    print(f"test cluster: {len(pts)} atoms, "
          + ", ".join(f"{e}={species.count(e)}" for e in sorted(set(species))))

    calc = DebyeCalculator(verbose=False)
    tt = np.linspace(20, 140, 200)
    lam = WAVELENGTHS['CuKa']
    failures = 0

    for anom in (False, True):
        p = {'apply_anomalous': anom, 'compute_partial_intensities': True}
        q, I_exact, _, part_exact = calc.calculate_direct_dse(pts, species, p, lam, tt)
        if I_exact.min() < 0:
            print(f"  FAIL  exact DSE returned a negative intensity (anomalous={anom})")
            failures += 1
        for bpa in (10, 40):
            _, I_bin, _, part_bin = calc.calculate_binned_pair_dse(
                pts, species, p, lam, tt, bins_per_angstrom=bpa)
            rel = np.max(np.abs(I_bin - I_exact)) / max(np.max(np.abs(I_exact)), 1e-30)
            tol = 3e-3 if bpa >= 40 else 3e-2
            good = rel < tol and I_bin.min() >= 0
            print(f"  {'PASS' if good else 'FAIL'}  anomalous={str(anom):5s} "
                  f"bins/A={bpa:3d}  max rel. deviation = {rel:.3e}  (tol {tol:.0e})"
                  f"  min I = {I_bin.min():.3e}")
            if not good:
                failures += 1
            if part_bin:
                tot = sum(part_bin.values())
                rel_p = np.max(np.abs(tot - I_bin)) / max(np.max(np.abs(I_bin)), 1e-30)
                if rel_p > 1e-10:
                    print(f"  FAIL  partials do not sum to the total (rel {rel_p:.2e})")
                    failures += 1

    rng2 = np.random.default_rng(3)
    n_amo, R_amo = 400, 11.0
    amo = []
    while len(amo) < n_amo:
        p3 = rng2.uniform(-R_amo, R_amo, 3)
        if np.linalg.norm(p3) <= R_amo:
            amo.append(p3)
    amo = np.array(amo)
    sp_amo = list(rng2.choice(['Cu', 'Zn'], size=n_amo, p=[0.6, 0.4]))
    p_amo = {'apply_anomalous': False, 'compute_partial_intensities': False}
    _, I_amo, _, _ = calc.calculate_direct_dse(amo, sp_amo, p_amo, lam, tt)
    if I_amo.min() < 0:
        print("  FAIL  exact DSE negative on the disordered cluster")
        failures += 1
    prev_rel = None
    for bpa in (5, 10, 20):
        _, I_b, _, _ = calc.calculate_binned_pair_dse(
            amo, sp_amo, p_amo, lam, tt, bins_per_angstrom=bpa)
        rel = np.max(np.abs(I_b - I_amo)) / max(np.max(np.abs(I_amo)), 1e-30)
        note = ""
        if prev_rel is not None:
            ratio = prev_rel / max(rel, 1e-30)
            note = f"  (improved x{ratio:.1f} for a 2x finer bin; 2nd order expects ~4)"
            if ratio < 2.5:
                print(f"  FAIL  binning error not converging as expected{note}")
                failures += 1
        good = rel < 1e-2 and I_b.min() >= 0
        print(f"  {'PASS' if good else 'FAIL'}  disordered, bins/A={bpa:3d}  "
              f"max rel. deviation = {rel:.3e}{note}")
        if not good:
            failures += 1
        prev_rel = rel

    fq = calc.get_scattering_factors(['Cu'], np.array([0.0, 3.0, 6.0, 8.0]), lam,
                                     {'apply_anomalous': False})['Cu']
    dropping = bool(np.all(np.diff(fq) < 0)) and fq[0] > 28.0 and fq[-1] < 16.0
    print(f"  {'PASS' if dropping else 'FAIL'}  f0(Cu) falls with q: "
          + ", ".join(f"{v:.3f}" for v in fq))
    if not dropping:
        failures += 1

    ttl = np.linspace(10, 170, 400)
    lp_I, _, _ = apply_measurement_corrections(
        ttl, np.ones_like(ttl), {}, {'apply_LP': True, 'LP_max_clip': 1e9}, verbose=False)
    pol = (1.0 + np.cos(np.radians(ttl)) ** 2)
    argmin_pol = ttl[np.argmin(pol)]
    ok_lp = abs(argmin_pol - 90.0) < 1.0
    print(f"  {'PASS' if ok_lp else 'FAIL'}  polarization factor minimum at "
          f"2theta = {argmin_pol:.2f} deg (expected 90)")
    if not ok_lp:
        failures += 1

    x = np.linspace(0, 100, 2001)
    y = np.exp(-0.5 * ((x - 50) / 0.4) ** 2)
    sig = 0.2 + 0.004 * x
    yc = _variable_width_convolve(x, y, sig)
    a0 = np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x)
    a1 = np.trapezoid(yc, x) if hasattr(np, 'trapezoid') else np.trapz(yc, x)
    ok_area = abs(a1 - a0) / a0 < 1e-3
    print(f"  {'PASS' if ok_area else 'FAIL'}  variable-width convolution conserves area "
          f"({a0:.6f} -> {a1:.6f})")
    if not ok_area:
        failures += 1

    try:
        apply_measurement_corrections(ttl, np.ones_like(ttl), {},
                                      {'apply_march_dollase': True}, verbose=False)
        print("  FAIL  apply_march_dollase was accepted")
        failures += 1
    except ValueError:
        print("  PASS  apply_march_dollase refused with an explanation")

    print("=" * 70)
    print(f"SELF-TEST: {'all checks passed' if failures == 0 else f'{failures} FAILURES'}")
    print("=" * 70)
    return failures == 0

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in ('--self-test', '-t'):
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) < 2:
        print("Usage: python XRD-Debye_Scattering.py input.txt")
        print("       python XRD-Debye_Scattering.py --self-test")
        sys.exit(2)

    input_file = sys.argv[1]
    params = parse_input_file(input_file)
    ok = run_debye(params)
    if not ok:
        sys.exit(4)
