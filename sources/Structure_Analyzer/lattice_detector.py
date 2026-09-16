# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

import argparse
import sys
import numpy as np
import os
from collections import Counter, defaultdict
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
import scipy.spatial.distance as distance
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import math
import warnings
import io


if sys.platform == "win32":
    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is not None and hasattr(_stream, "buffer"):
            try:
                setattr(sys, _name, io.TextIOWrapper(
                    _stream.buffer, encoding="utf-8",
                    errors="replace", line_buffering=True))
            except Exception:
                pass

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

try:
    import ovito
    from ovito.io import import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    from ovito.data import DataCollection
    OVITO_AVAILABLE = True
except ImportError:
    OVITO_AVAILABLE = False
    print("Warning: OVITO not available. Using fallback methods.")


class AnalysisParameters:
    def __init__(self):
        self.neighbor_k = 20
        self.rmsd_cutoff = 0.1
        self.verbose = True
        self.use_advanced_methods = True
        self.plot = False

        self.vector_stats_min_neighbors = 6
        self.vector_stats_max_cutoff = 5.0
        self.vector_stats_tolerance_factor = 0.12

        self.vector_stats_n_trials = 200
        self.vector_stats_sample_size = 3000
        self.vector_stats_hist_bins = 300
        self.vector_stats_random_seed = 42

        self.vector_stats_k_nn = 20          
        self.vector_stats_dbscan_eps = 0.07   
        self.vector_stats_merge_threshold = 0.85

        self.ls_sample_size = 2000
        self.ls_bin_width = 0.01
        self.ls_k_neighbors = 6
        self.ls_use_periodic = None


    KEY_ALIASES = {
        'neighbour_k': 'neighbor_k',
        'ls_k_neighbours': 'ls_k_neighbors',
        'vector_stats_min_neighbours': 'vector_stats_min_neighbors',
        'vector_stats_k_nn_neighbours': 'vector_stats_k_nn',
    }

    NON_PARAM_KEYS = {'input', 'output', 'output_dir'}

    @classmethod
    def from_dict(cls, param_dict):
        params = cls()

        unknown = []
        for raw_key, value in param_dict.items():
            key = cls.KEY_ALIASES.get(raw_key, raw_key)
            if not hasattr(params, key) and raw_key not in cls.NON_PARAM_KEYS:
                unknown.append(raw_key)
            if hasattr(params, key):
                if key == 'ls_use_periodic':
                    if value is None or str(value).strip().lower() in ['none', 'null', '']:
                        setattr(params, key, None)
                    elif str(value).strip().lower() in ['true', 'yes', '1', 'on']:
                        setattr(params, key, True)
                    elif str(value).strip().lower() in ['false', 'no', '0', 'off']:
                        setattr(params, key, False)
                    else:
                        setattr(params, key, None) 
                    continue

                if isinstance(value, str):
                    value = value.strip()
                    if value.lower() in ['on', 'true', 'yes', '1']:
                        value = True
                    elif value.lower() in ['off', 'false', 'no', '0']:
                        value = False
                    elif value.lower() == 'none':
                        value = None
                    elif '.' in value:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            pass

                setattr(params, key, value)

        if unknown:
            print("Warning: unrecognised parameter(s) ignored: "
                  + ", ".join(sorted(unknown)))
            print("         (check spelling; these had no effect on the analysis)")

        return params

def read_parameter_file(filepath):
    params = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().split('#')[0].strip()  # Remove comments after #

                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    if value.lower() == 'none':
                        value = None
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False

                    params[key] = value

        return params
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        return None


def plot_distance_histogram(distances, peaks, bin_edges, title, filename, char_dists=None):
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.figure(figsize=(12, 8))

    plt.hist(distances, bins=bin_edges, alpha=0.7, color='skyblue', edgecolor='black', density=True)

    for i, peak in enumerate(peaks[:10]):  
        peak_pos = (bin_edges[peak] + bin_edges[peak+1]) / 2
        plt.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

        if i < 5:  
            plt.text(peak_pos, plt.ylim()[1]*0.95, f'd{i+1}: {peak_pos:.3f} Å',
                    rotation=90, verticalalignment='top', fontsize=9)

    plt.xlabel('Distance (Å)', fontsize=14)
    plt.ylabel('Normalized Frequency', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    stats_text = f'Total distances: {len(distances):,}\n'
    stats_text += f'Mean: {np.mean(distances):.3f} Å\n'
    stats_text += f'Std: {np.std(distances):.3f} Å'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {filename}")

def plot_3d_vectors(positions, basis_vectors, filename, sample_size=500,
                    random_seed=0):
    if not MATPLOTLIB_AVAILABLE:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if len(positions) > sample_size:
        indices = np.random.RandomState(int(random_seed)).choice(
            len(positions), sample_size, replace=False)
        sample_pos = positions[indices]
    else:
        sample_pos = positions

    ax.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2],
               c='lightgray', alpha=0.4, s=5, label=f'Atoms (sample of {len(sample_pos)})')

    center = np.mean(sample_pos, axis=0)

    colors = ['red', 'green', 'blue']
    labels = ['a vector', 'b vector', 'c vector']
    vector_lengths = []

    scale_factor = 3.0

    for i, (vec, color, label) in enumerate(zip(basis_vectors, colors, labels)):
        vec_length = np.linalg.norm(vec)
        vector_lengths.append(vec_length)

        scaled_vec = vec * scale_factor

        ax.quiver(center[0], center[1], center[2],
                  scaled_vec[0], scaled_vec[1], scaled_vec[2],
                  color=color, arrow_length_ratio=0.15, linewidth=3,
                  label=f'{label}: {vec_length:.3f} Å', alpha=0.8)

        end_point = center + scaled_vec
        ax.text(end_point[0], end_point[1], end_point[2],
                f'{label[0].upper()}', fontsize=12, fontweight='bold', color=color)

    max_length = max(vector_lengths) * scale_factor
    ax.set_xlim([center[0] - max_length, center[0] + max_length])
    ax.set_ylim([center[1] - max_length, center[1] + max_length])
    ax.set_zlim([center[2] - max_length, center[2] + max_length])

    ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
    ax.set_title('Lattice Vectors Visualization (vectors scaled for visibility)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)

    info_text = f'Vector scale factor: {scale_factor}x\n'
    info_text += f'View centered on: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}'
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  3D plot saved: {filename}")
    print(f"  Actual vector lengths: a={vector_lengths[0]:.3f} Å, " +
          f"b={vector_lengths[1]:.3f} Å, c={vector_lengths[2]:.3f} Å")

def plot_comparison_histograms(vector_lengths, nn_distances, filename):

    vector_lengths = np.asarray(vector_lengths, dtype=float)
    nn_distances = np.asarray(nn_distances, dtype=float)
    if len(vector_lengths) == 0 or len(nn_distances) == 0:
        return

    import json
    json_path = os.path.join(os.path.dirname(os.path.abspath(filename)),
                             'lattice_hist_comp_data.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump({'vector_lengths': vector_lengths.tolist(),
                       'pairwise_distances': nn_distances.tolist()}, jf)
        print(f"  Plot data saved: {json_path}")
    except Exception as e:
        print(f"  Warning: could not write plot data JSON: {e}")

    if not MATPLOTLIB_AVAILABLE:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(vector_lengths, bins=100, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax1.set_xlabel('Vector Length (Å)', fontsize=12)
    ax1.set_ylabel('Normalized Frequency', fontsize=12)
    ax1.set_title('Interatomic Vector Lengths\n(Vector Statistics Method)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Count: {len(vector_lengths):,}', transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.hist(nn_distances, bins=100, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    ax2.set_xlabel('Nearest-Neighbour Distance (Å)', fontsize=12)
    ax2.set_ylabel('Normalized Frequency', fontsize=12)
    ax2.set_title('Nearest-Neighbour Distances\n(RDF Shell-Ratio Method)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Count: {len(nn_distances):,}', transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Comparison plot saved: {filename}")


def _plot_comparison_if_requested(params, vector_lengths_data, nn_distances_data):
    if not params.plot:
        pass
    elif not MATPLOTLIB_AVAILABLE:
        print("  Plotting skipped: matplotlib not available.")
    elif vector_lengths_data is None:
        print("  Plotting skipped: vector statistics data missing.")
    elif nn_distances_data is None:
        print("  Plotting skipped: nearest-neighbour data missing.")
    else:
        plot_comparison_histograms(
            vector_lengths_data,
            nn_distances_data,
            "comparison_histograms.png"
        )
        print("  Comparison histogram saved: comparison_histograms.png")


try:
    from sklearn.cluster import DBSCAN, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Vector statistics will use simpler clustering.")

def _gauss(x, A, mu, sigma, bg):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

def extract_lattice_by_vector_statistics_improved(
        positions,
        dominant_structure=None,
        params=None,
        verbose=True,
        random_seed=0):

    rng = np.random.RandomState(random_seed)

    defaults = {
        'vector_stats_min_neighbors': 6,
        'vector_stats_max_cutoff': 5.0,
        'vector_stats_k_nn': getattr(params, 'neighbor_k', 20) if params else 20,
        'vector_stats_tolerance_factor': 0.12,
        'vector_stats_n_trials': 200,
        'vector_stats_sample_size': 3000,
        'vector_stats_hist_bins': 300,
        'plot': False,
    }
    if params is None:
        P = defaults
    else:
        P = defaults.copy()
        P.update(vars(params) if hasattr(params, '__dict__') else params)

    eps_val = float(P.get('vector_stats_dbscan_eps', 0.07))
    min_samples_val = int(P.get('vector_stats_min_neighbors', 8))
    merge_thresh = float(P.get('vector_stats_merge_threshold', 0.85))

    N = len(positions)
    if N < 6:
        raise ValueError("Need at least ~6 atoms; got %d" % N)

    if verbose:
        print("[vector_statistics] N=", N, "params:", {k:P[k] for k in P})


    sample_size = min(int(P['vector_stats_sample_size']), N)
    sample_idx = rng.choice(N, sample_size, replace=False)
    sample_positions = positions[sample_idx]


    tree = cKDTree(positions)


    k_val = int(P.get('vector_stats_k_nn', 20))
    _cut_raw = P.get('vector_stats_max_cutoff', 5.0)

    _probe_d, _ = tree.query(sample_positions[:min(400, len(sample_positions))], k=2)
    _d_probe = float(np.median(_probe_d[:, 1]))


    if isinstance(_cut_raw, str) and _cut_raw.strip().lower() == 'auto':
        max_cutoff = 1.45 * _d_probe
        if verbose:
            print("  vector_stats_max_cutoff=auto -> {:.3f} A "
                  "(1.45 x d_nn probe {:.3f} A)".format(max_cutoff, _d_probe))
    else:
        max_cutoff = float(_cut_raw)
        if max_cutoff < 1.15 * _d_probe:
            print("  WARNING: vector_stats_max_cutoff = {:.2f} A is below "
                  "1.15 x the nearest-neighbour distance ({:.2f} A); the first "
                  "coordination shell will be truncated. Consider 'auto' or "
                  ">= {:.2f} A.".format(max_cutoff, _d_probe, 1.45 * _d_probe))
        elif verbose and max_cutoff > 3.5 * _d_probe:
            print("  NOTE: vector_stats_max_cutoff = {:.2f} A is more than "
                  "3.5 x d_nn ({:.2f} A); direction clustering may pick up "
                  "far shells.".format(max_cutoff, _d_probe))


    dists, neighbors_list = tree.query(sample_positions, k=k_val + 1)

    all_vecs = []
    lengths = []

    for local_i, (neigh_dists, neigh_inds) in enumerate(zip(dists, neighbors_list)):
        pi = sample_positions[local_i]

        for d, j_idx in zip(neigh_dists, neigh_inds):
            if j_idx >= len(positions) or d < 1e-6:
                continue

            if d > max_cutoff:
                continue

            if d < 0.5:
                continue


            v = positions[j_idx] - pi

            all_vecs.append(v)
            lengths.append(d)

    if len(lengths) == 0:
        if verbose:
            print("No interatomic vectors found within cutoff.")
        return None, {'error': 'no_vectors'}

    lengths = np.array(lengths)
    all_vecs = np.array(all_vecs)

    popt = None
    centers = None
    hist = None

    nn_dists, _ = tree.query(sample_positions, k=2)  
    local_nn = nn_dists[:, 1]
    d_nn_median = float(np.median(local_nn))
    sigma_nn_median = float(np.std(local_nn))

    rdf_shells, rdf_diag = compute_rdf_shells(
        positions, sample_size=min(4000, N), rng=rng, verbose=False)
    if rdf_shells:
        d1 = float(rdf_shells[0])
        _in_shell = np.abs(lengths - d1) <= 0.10 * d1
        sigma1 = float(lengths[_in_shell].std()) if _in_shell.sum() > 10 else sigma_nn_median
        d1_source = 'rdf_first_shell'
    else:
        d1 = d_nn_median
        sigma1 = sigma_nn_median
        d1_source = 'nn_median_fallback'
        if verbose:
            print("  RDF gave no shells; falling back to median NN distance "
                  "(biased low for disordered input).")

    if verbose:
        print("First-shell distance d1 = {:.6f} A (sigma {:.6f}, from {})".format(
            d1, sigma1, d1_source))
        print("  [diagnostic] median per-atom NN = {:.6f} A".format(d_nn_median))


    hist, edges = np.histogram(lengths, bins=int(P['vector_stats_hist_bins']))
    centers = 0.5 * (edges[:-1] + edges[1:])
    popt = None 

    if verbose and rdf_shells:
        print("  RDF shells (A): " +
              ", ".join("%.4f" % x for x in rdf_shells))


    tol = max( P['vector_stats_tolerance_factor'] * d1, max(0.02, sigma1*0.8) )
    mask = np.abs(lengths - d1) <= tol
    cand_vecs = all_vecs[mask]
    cand_lens = lengths[mask]
    if len(cand_vecs) < 60:
        mask2 = np.abs(lengths - d1) <= max(1.5*tol, 0.2)
        cand_vecs = all_vecs[mask2]
        cand_lens = lengths[mask2]
    if len(cand_vecs) < 30:
        if verbose: print(f"Too few candidate NN vectors ({len(cand_vecs)}). Aborting.")
        return None, {'error':'too_few_candidates', 'num_candidates':len(cand_vecs)}

    unit_dirs = cand_vecs / np.linalg.norm(cand_vecs, axis=1)[:,None]

    labels = None
    if SKLEARN_AVAILABLE:

        db = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(unit_dirs)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 3:

            km = KMeans(n_clusters=6, random_state=random_seed).fit(unit_dirs)
            labels = km.labels_
    else:
        k = 6
        centers_k = unit_dirs[rng.choice(len(unit_dirs), k, replace=False)]
        for _ in range(10):
            dists = np.dot(unit_dirs, centers_k.T) 
            assign = np.argmax(dists, axis=1)
            for j in range(k):
                members = unit_dirs[assign==j]
                if len(members)>0:
                    centers_k[j] = np.mean(members, axis=0)
                    centers_k[j] /= np.linalg.norm(centers_k[j]) + 1e-12
        labels = assign

    unique_labels = [lab for lab in sorted(set(labels)) if lab != -1]
    clusters = []
    for lab in unique_labels:
        members = unit_dirs[labels==lab]
        count = len(members)
        if count == 0: continue
        center = np.mean(members, axis=0)
        center /= np.linalg.norm(center) + 1e-12
        clusters.append({'label':lab, 'count':count, 'center':center})
    clusters = sorted(clusters, key=lambda c: c['count'], reverse=True)

    if verbose:
        print(f"Found {len(clusters)} direction clusters (taking top 6):", [c['count'] for c in clusters[:6]])

    dir_centers = []
    used = []
    for c in clusters:
        vec = c['center'] * d1
        if any(abs(np.dot(vec, u)) / (np.linalg.norm(vec)*np.linalg.norm(u) + 1e-12) > merge_thresh for u in used):
            continue
        used.append(vec)
        dir_centers.append((vec, c['count']))
        if len(dir_centers) >= 6:
            break
    best_trip = None
    best_score = -1.0
    M = len(dir_centers)
    for i in range(M):
        for j in range(i+1, M):
            for k in range(j+1, M):
                v1 = dir_centers[i][0]
                v2 = dir_centers[j][0]
                v3 = dir_centers[k][0]
                det = abs(np.linalg.det(np.vstack([v1, v2, v3])))
                if det < 1e-6:
                    continue
                pop_score = dir_centers[i][1] + dir_centers[j][1] + dir_centers[k][1]
                score = det * (1.0 + 0.02 * pop_score)
                if score > best_score:
                    best_score = score
                    best_trip = (v1, v2, v3)

    if best_trip is None:
        if verbose: print("No independent triplet found - try relaxing parameters.")
        return None, {'error':'no_independent_triplet'}

    v1, v2, v3 = best_trip

    Mmat = canonicalize_primitive(np.vstack([v1, v2, v3])).T  # 3x3, columns
    if np.linalg.det(Mmat) < 0:
        Mmat[:, 0] *= -1.0

    struct = (dominant_structure or '').upper()
    a_conventional = (float(d1 * NN_TO_CONVENTIONAL_A[struct])
                      if struct in NN_TO_CONVENTIONAL_A else None)

    a = np.linalg.norm(Mmat[:,0])
    b = np.linalg.norm(Mmat[:,1])
    c = np.linalg.norm(Mmat[:,2])
    def angle_deg(u,v):
        cosang = np.dot(u,v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12)
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))
    alpha = angle_deg(Mmat[:,1], Mmat[:,2])
    beta  = angle_deg(Mmat[:,0], Mmat[:,2])
    gamma = angle_deg(Mmat[:,0], Mmat[:,1])

    orthogonality_score = 3.0 - (abs(np.dot(Mmat[:,0], Mmat[:,1])/(a*b))+abs(np.dot(Mmat[:,0], Mmat[:,2])/(a*c))+abs(np.dot(Mmat[:,1], Mmat[:,2])/(b*c)))
    vol = abs(np.linalg.det(Mmat))

    result = {
        'structure': dominant_structure or 'UNKNOWN',
        'method': 'vector_statistics',
        'a_primitive': float(a), 'b_primitive': float(b), 'c_primitive': float(c),
        'alpha_primitive': float(alpha), 'beta_primitive': float(beta),
        'gamma_primitive': float(gamma),
        'primitive_vectors': [Mmat[:, 0].tolist(), Mmat[:, 1].tolist(),
                              Mmat[:, 2].tolist()],
        'volume_primitive': float(vol),

        'd1': float(d1),
        'd1_source': d1_source,
        'd1_sigma': float(sigma1),
        'd_nn_median': float(d_nn_median),
        'num_candidate_vectors': int(len(cand_vecs)),
        'num_clusters': int(len(clusters)),
        'orthogonality_score': float(orthogonality_score),
        'best_score': float(best_score),
    }

    if struct in ('FCC', 'BCC') and a_conventional:
        conv = conventional_from_primitive(Mmat.T, struct)
        if conv is not None:
            (ca_, cb_, cc_), (al_, be_, ga_), Cvec = conv
            max_len_dev = (max(ca_, cb_, cc_) - min(ca_, cb_, cc_)) / max(ca_, cb_, cc_)
            max_ang_dev = max(abs(al_ - 90.0), abs(be_ - 90.0), abs(ga_ - 90.0))
            is_cubic = (max_len_dev < 0.02 and max_ang_dev < 1.5)
            result.update({
                'a': float(ca_), 'b': float(cb_), 'c': float(cc_),
                'alpha': al_, 'beta': be_, 'gamma': ga_,
                'cell_type': ('conventional_cubic' if is_cubic
                              else 'conventional_distorted'),
                'conventional_vectors': Cvec.tolist(),
                'volume': float(abs(np.linalg.det(Cvec))),
                'a_from_d1': a_conventional,
                'cubic_consistent': bool(is_cubic),
                'axial_ratio_c_over_a': float(cc_ / ca_) if ca_ > 0 else None,
                'max_length_deviation': float(max_len_dev),
                'max_angle_deviation_deg': float(max_ang_dev),
            })
            if not is_cubic:
                result['note'] = (
                    'Cell is NOT cubic within tolerance (lengths differ by '
                    '{:.1%}, angles deviate up to {:.2f} deg). The structure '
                    'label "{}" came from PTM/the caller; the geometry measured '
                    'here is distorted - check for tetragonal (Bain-path) or '
                    'sheared states.'.format(max_len_dev, max_ang_dev, struct))
                if verbose:
                    print("  WARNING: " + result['note'])
        else:
            result.update({
                'a': a_conventional, 'b': a_conventional, 'c': a_conventional,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
                'cell_type': 'conventional_cubic',
                'volume': float(a_conventional ** 3),
                'cubic_consistent': None,
            })
    elif struct == 'HCP' and a_conventional:
        _c, _c_src = hcp_c_from_shells(a_conventional, rdf_shells)
        result.update({
            'a': a_conventional, 'b': a_conventional,
            'c': (float(_c) if _c else float(HCP_IDEAL_C_OVER_A * a_conventional)),
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'cell_type': 'conventional_hexagonal',
            'c_source': _c_src,
        })
        result['c_over_a'] = float(result['c'] / result['a'])
        result['volume'] = float(result['a'] ** 2 * result['c'] *
                                 math.sin(math.radians(120.0)))
    else:
        result.update({
            'a': float(a), 'b': float(b), 'c': float(c),
            'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
            'cell_type': 'primitive',
            'volume': float(vol),
        })

    diagnostics = {
        'lengths_hist_centers': centers,
        'lengths_hist': hist,
        'fit_popt': popt,
        'rdf': rdf_diag,
        'rdf_shells': rdf_shells,
        'unit_direction_clusters': clusters,
        'candidate_vectors': cand_vecs,
        'all_vectors_count': int(len(lengths)),
        'lengths': lengths,                 # REQUIRED for plotting
        'all_vector_lengths': lengths       # optional alias (defensive)
    }


    if verbose:
        print("  Reported cell ({}): a={:.4f} b={:.4f} c={:.4f} A, "
              "alpha={:.2f} beta={:.2f} gamma={:.2f} deg, V={:.3f} A^3".format(
                  result['cell_type'], result['a'], result['b'], result['c'],
                  result['alpha'], result['beta'], result['gamma'],
                  result['volume']))
        print("  Primitive cell as measured: |v|=({:.4f}, {:.4f}, {:.4f}) A, "
              "angles=({:.2f}, {:.2f}, {:.2f}) deg, V={:.3f} A^3".format(
                  a, b, c, alpha, beta, gamma, vol))
        print("  Direction clusters:",
              [(cc['count'], np.round(cc['center'], 3).tolist()) for cc in clusters[:6]])
    return result, diagnostics




def _is_valid_cell(cell):

    if cell is None:
        return False
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return False
    # volume
    vol = np.dot(cell[0], np.cross(cell[1], cell[2]))
    if abs(vol) < 1e-8:
        return False
    return True

def compute_nn_distribution(positions, cell=None, k=6, sample_size=2000,
                            periodic=None, random_seed=0):

    positions = np.asarray(positions, dtype=float)
    N = len(positions)
    if N < 2:
        return np.array([])

    if periodic is None:
        use_periodic = _is_valid_cell(cell)
    else:
        use_periodic = bool(periodic) and _is_valid_cell(cell)

    rng = np.random.RandomState(int(random_seed))
    if N > sample_size:
        idx = rng.choice(N, int(sample_size), replace=False)
    else:
        idx = np.arange(N)
    query_pts = positions[idx]

    if not use_periodic:
        tree = cKDTree(positions)

        k_query = min(int(k) + 1, N)
        dists, _ = tree.query(query_pts, k=k_query)
        dists = np.atleast_2d(dists)
        valid = dists > 1e-12
        has_any = valid.any(axis=1)
        first = np.argmax(valid, axis=1)
        nn = dists[np.arange(len(dists)), first]
        nn = nn[has_any & np.isfinite(nn)]
        return nn

    cell = np.asarray(cell, dtype=float)
    offsets = np.array([[i, j, m]
                        for i in (-1, 0, 1) for j in (-1, 0, 1) for m in (-1, 0, 1)],
                       dtype=float)
    shifts = offsets @ cell                      # 27 x 3
    reps = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

    tree = cKDTree(reps)
    k_query = min(int(k) + 1, len(reps))
    dists, _ = tree.query(query_pts, k=k_query)
    dists = np.atleast_2d(dists)
    valid = dists > 1e-12
    has_any = valid.any(axis=1)
    first = np.argmax(valid, axis=1)
    nn = dists[np.arange(len(dists)), first]
    nn = nn[has_any & np.isfinite(nn)]
    return nn


def fit_first_peak(nn_array, bin_width=0.01):
    if len(nn_array) < 8:
        return None
    median = np.median(nn_array)
    lo = max(0.0, median * 0.6)
    hi = median * 1.6
    nbins = max(30, int((hi - lo) / bin_width))
    hist, edges = np.histogram(nn_array, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.max(hist) <= 0:
        return None
    min_height = np.max(hist) * 0.05
    peaks, props = find_peaks(hist, height=min_height, distance=3)
    if len(peaks) == 0:
        return None
    p = peaks[np.argmax(hist[peaks])]
    init_mu = centers[p]
    init_A = hist[p]
    init_sigma = max(bin_width, init_mu * 0.01)
    init_bg = np.min(hist)
    try:
        popt, pcov = curve_fit(_gauss, centers, hist, p0=[init_A, init_mu, init_sigma, init_bg], maxfev=5000)
        A, mu, sigma, bg = popt
        return dict(mu=float(mu), sigma=float(abs(sigma)), A=float(A), bg=float(bg), hist_x=centers, hist_y=hist)
    except Exception:
        return dict(mu=float(median), sigma=float(np.std(nn_array)), A=float(np.max(hist)), bg=float(np.min(hist)), hist_x=centers, hist_y=hist)
        
STRUCTURE_SHELL_RATIOS = {
    'FCC': (1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0),
    'BCC': (1.0, 2.0 / np.sqrt(3.0), 2.0 * np.sqrt(2.0) / np.sqrt(3.0),
            np.sqrt(11.0) / np.sqrt(3.0)),
    'HCP': (1.0, np.sqrt(2.0), np.sqrt(8.0 / 3.0), np.sqrt(3.0)),
}

NN_TO_CONVENTIONAL_A = {
    'FCC': np.sqrt(2.0),        
    'BCC': 2.0 / np.sqrt(3.0),   
    'HCP': 1.0,                  
}

HCP_IDEAL_C_OVER_A = np.sqrt(8.0 / 3.0)


def expected_distances_for_structure(a, structure):
    s = (structure or '').upper()
    ratios = STRUCTURE_SHELL_RATIOS.get(s)
    if ratios is None:
        return []
    d1 = a / NN_TO_CONVENTIONAL_A.get(s, 1.0)
    return [d1 * r for r in ratios]


def score_structure_from_char(char_dists, structure):

    if char_dists is None or len(char_dists) < 1:
        return None, None
    s = (structure or '').upper()
    d1 = float(char_dists[0])
    if d1 <= 0:
        return None, None
    a_cand = float(d1 * NN_TO_CONVENTIONAL_A.get(s, 1.0))

    ideal = STRUCTURE_SHELL_RATIOS.get(s)
    if ideal is None:
        return None, a_cand

    measured = np.asarray(char_dists, dtype=float) / d1
    n = min(len(measured) - 1, len(ideal) - 1)   
    if n < 1:

        return None, a_cand

    resid = [(measured[i] - ideal[i]) / ideal[i] for i in range(1, n + 1)]
    return float(np.sqrt(np.mean(np.square(resid)))), a_cand

STRUCTURE_ACCEPT_ERR = 0.06


def classify_structure_from_shells(char_dists, candidates=('FCC', 'BCC', 'HCP'),
                                   accept_err=STRUCTURE_ACCEPT_ERR):

    scores = {}
    for st in candidates:
        err, _ = score_structure_from_char(char_dists, st)
        if err is not None:
            scores[st] = err
    if not scores:
        return None, {}, False

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_err = ranked[0]

    if best_err > accept_err:
        return None, scores, True

    ambiguous = False
    if len(ranked) > 1:
        second_err = ranked[1][1]
        if second_err > 0 and (second_err - best_err) / second_err < 0.25:
            ambiguous = True
    if best in ('FCC', 'HCP') and {'FCC', 'HCP'} <= set(scores):
        ambiguous = True

    return best, scores, ambiguous


def _cell_angles(v1, v2, v3):
    def ang(u, v):
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return ang(v2, v3), ang(v1, v3), ang(v1, v2)


def canonicalize_primitive(vectors):

    import itertools
    V0 = np.asarray(vectors, dtype=float)
    v_prim = abs(np.linalg.det(V0))
    if v_prim < 1e-12:
        return V0

    cands = []
    seen = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                if (n1, n2, n3) == (0, 0, 0) or (n1, n2, n3) < (0, 0, 0):
                    continue
                v = n1 * V0[0] + n2 * V0[1] + n3 * V0[2]
                if np.dot(v, v) < 1e-12:
                    continue
                if any(np.allclose(v, w, atol=1e-8) for w in seen):
                    continue
                seen.append(v)
                cands.append(v)
    cands.sort(key=lambda v: float(np.dot(v, v)))
    cands = cands[:30]

    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            for k in range(j + 1, len(cands)):
                C = np.array([cands[i], cands[j], cands[k]])
                det = abs(np.linalg.det(C))
                if det < 1e-12 or abs(det - v_prim) / v_prim > 1e-3:
                    continue
                len2 = float(np.sum([np.dot(v, v) for v in C]))
                ang = _cell_angles(C[0], C[1], C[2])
                spread = max(ang) - min(ang)
                rank = (round(len2, 6), round(spread, 4))
                if best is None or rank < best[0]:
                    best = (rank, C)
    V = best[1] if best is not None else V0

    best_s = None
    tol = 1e-9
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            p = np.array([signs[t] * V[perm[t]] for t in range(3)])
            if np.linalg.det(p) <= 1e-9:          # keep right-handed only
                continue
            dots = (float(np.dot(p[0], p[1])),
                    float(np.dot(p[0], p[2])),
                    float(np.dot(p[1], p[2])))
            type_i = all(d >= -tol for d in dots)
            type_ii = all(d <= tol for d in dots)
            if not (type_i or type_ii):
                continue
            ang = _cell_angles(p[0], p[1], p[2])
            rank = (0 if type_i else 1,
                    tuple(round(float(np.linalg.norm(x)), 8) for x in p),
                    tuple(round(x, 6) for x in ang))
            if best_s is None or rank < best_s[0]:
                best_s = (rank, p)
    return best_s[1] if best_s is not None else V


def conventional_from_primitive(vectors, structure):

    st = (structure or '').upper()
    n_prim = {'FCC': 4, 'BCC': 2}.get(st)
    if n_prim is None:
        return None
    P = np.asarray(vectors, dtype=float)
    v_prim = abs(np.linalg.det(P))
    if v_prim < 1e-12:
        return None
    target = n_prim * v_prim

    cands = []
    seen = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                if (n1, n2, n3) == (0, 0, 0):
                    continue
                if (n1, n2, n3) < (0, 0, 0):      # skip the antipode
                    continue
                v = n1 * P[0] + n2 * P[1] + n3 * P[2]
                if np.dot(v, v) < 1e-12:
                    continue
                if any(np.allclose(v, w, atol=1e-8) for w in seen):
                    continue
                seen.append(v)
                cands.append(v)
    cands.sort(key=lambda v: float(np.dot(v, v)))
    cands = cands[:40]

    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            for k in range(j + 1, len(cands)):
                C = np.array([cands[i], cands[j], cands[k]])
                det = abs(np.linalg.det(C))
                if det < 1e-12 or abs(det - target) / target > 1e-3:
                    continue
                angles = _cell_angles(C[0], C[1], C[2])
                ang_score = sum(abs(x - 90.0) for x in angles)
                len_score = float(np.sum([np.dot(v, v) for v in C]))
                rank = (round(ang_score, 6), round(len_score, 6))
                if best is None or rank < best[0]:
                    best = (rank, C, angles)
    if best is None:
        return None

    _rank, C, angles = best
    if np.linalg.det(C) < 0:
        C = np.array([C[0], C[2], C[1]])
        angles = _cell_angles(C[0], C[1], C[2])
    # Order axes so that any unique (tetragonal) axis is reported as c.
    lengths = [float(np.linalg.norm(v)) for v in C]
    order = sorted(range(3), key=lambda t: lengths[t])
    if abs(lengths[order[0]] - lengths[order[1]]) < abs(lengths[order[1]] - lengths[order[2]]):
        perm = [order[0], order[1], order[2]]      # unique axis is the long one
    else:
        perm = [order[1], order[2], order[0]]      # unique axis is the short one
    C = np.array([C[perm[0]], C[perm[1]], C[perm[2]]])
    if np.linalg.det(C) < 0:
        C[2] = -C[2]
    angles = _cell_angles(C[0], C[1], C[2])
    lengths = tuple(float(np.linalg.norm(v)) for v in C)
    return lengths, tuple(float(x) for x in angles), C


def _smooth(y, width):
    if width < 2:
        return np.asarray(y, dtype=float)
    kern = np.ones(int(width), dtype=float) / float(int(width))
    return np.convolve(np.asarray(y, dtype=float), kern, mode='same')


def _periodic_images(positions, cell):

    if not _is_valid_cell(cell):
        return np.asarray(positions, dtype=float), 1
    cell = np.asarray(cell, dtype=float)
    offsets = np.array([[i, j, k]
                        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                       dtype=float)
    shifts = offsets @ cell
    reps = (np.asarray(positions, dtype=float)[None, :, :]
            + shifts[:, None, :]).reshape(-1, 3)
    return reps, len(shifts)


def compute_rdf_shells(positions, rmax=None, bin_width=0.02, sample_size=4000,
                       n_shells=4, rng=None, verbose=False, cell=None):
    positions = np.asarray(positions, dtype=float)
    N = len(positions)
    if N < 20:
        return [], {'error': 'too_few_atoms'}
    if rng is None:
        rng = np.random.RandomState(0)
    periodic = _is_valid_cell(cell)
    neighbor_positions, _n_img = _periodic_images(positions, cell) if periodic \
        else (positions, 1)
    tree = cKDTree(neighbor_positions)

    probe = positions[rng.choice(N, min(400, N), replace=False)]
    d_probe, _ = tree.query(probe, k=2)
    d_nn_probe = float(np.median(d_probe[:, 1]))
    if not np.isfinite(d_nn_probe) or d_nn_probe <= 0:
        return [], {'error': 'degenerate_positions'}
    if rmax is None:
        rmax = max(3.0, 2.7 * d_nn_probe)

    n_sample = min(int(sample_size), N)
    idx = rng.choice(N, n_sample, replace=False)

    if periodic:
        interior = idx
    else:
        counts = np.asarray(tree.query_ball_point(positions[idx], rmax,
                                                  return_length=True))
        med = float(np.median(counts))
        interior = idx[counts >= 0.90 * med] if med > 0 else idx
        if len(interior) < 50:
            interior = idx

    nbins = max(50, int(np.ceil(rmax / max(bin_width, 1e-6))))
    edges = np.linspace(0.0, rmax, nbins + 1)
    counts_hist = np.zeros(nbins, dtype=np.float64)

    for start in range(0, len(interior), 512):
        chunk = positions[interior[start:start + 512]]
        for q, nb in zip(chunk, tree.query_ball_point(chunk, rmax)):
            r = np.linalg.norm(neighbor_positions[nb] - q, axis=1)
            r = r[r > 1e-9]
            if r.size:
                counts_hist += np.histogram(r, bins=edges)[0]

    centers = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(centers > 0, counts_hist / (centers ** 2), 0.0)
    if g.max() <= 0:
        return [], {'error': 'empty_rdf'}
    g = g / g.max()
    g_smooth = _smooth(g, max(2, int(0.03 * d_nn_probe / bin_width)))
    peaks, _ = find_peaks(g_smooth, height=0.03, prominence=0.015,
                          distance=max(2, int(0.09 * d_nn_probe / bin_width)))
    half = max(2, int(0.10 * d_nn_probe / bin_width))
    shells = []
    for p in peaks:
        lo, hi = max(0, p - half), min(nbins, p + half + 1)
        wt = counts_hist[lo:hi]
        if wt.sum() <= 0:
            continue
        shells.append(float(np.sum(centers[lo:hi] * wt) / wt.sum()))
        if len(shells) >= n_shells:
            break

    if verbose:
        print("  RDF: rmax=%.2f A, centres=%d, shells=%s"
              % (rmax, len(interior), [round(x, 4) for x in shells]))
        if len(shells) > 1:
            print("       measured ratios: " +
                  ", ".join("%.4f" % (x / shells[0]) for x in shells))

    return shells, {'r': centers, 'g': g, 'counts': counts_hist,
                    'rmax': float(rmax), 'n_centers': int(len(interior)),
                    'periodic': bool(periodic)}

def hcp_c_from_shells(a, shells, rel_tol=0.03):
    if a is None or a <= 0 or not shells:
        return None, 'unavailable'

    in_plane = (np.sqrt(2.0), np.sqrt(3.0), 2.0)
    lo, hi = 1.50, 1.95

    candidates = []
    for d in shells:
        r = d / a
        if not (lo <= r <= hi):
            continue
        if any(abs(r - ip) / ip <= rel_tol for ip in in_plane):
            continue
        candidates.append(float(d))

    if len(candidates) == 1:
        return candidates[0], 'c_axis_shell'
    return float(HCP_IDEAL_C_OVER_A * a), 'assumed_ideal'


def extract_lattice_auto(positions, dominant_structure=None, cell=None,
                         periodic=None, sample_size=2000, bin_width=0.01, k=6,
                         verbose=True, random_seed=0):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be Nx3 array")

    rng = np.random.RandomState(int(random_seed))

    if periodic is None:
        use_periodic = _is_valid_cell(cell)
    else:
        use_periodic = bool(periodic) and _is_valid_cell(cell)
    if verbose:
        if use_periodic:
            print("Periodic cell detected/used for NN (PBC ON).")
        else:
            print("No valid cell detected; running in NON-PERIODIC mode (PBC OFF).")

    nn = compute_nn_distribution(positions, cell=cell if use_periodic else None,
                                 periodic=use_periodic, sample_size=sample_size,
                                 k=k, random_seed=random_seed)
    if len(nn) < 8:
        if verbose:
            print("Insufficient NN samples (<8). Aborting.")
        return None, {'nn_array': nn}

    shells, rdf_diag = compute_rdf_shells(positions, sample_size=max(2000, sample_size),
                                          bin_width=max(0.005, min(0.05, bin_width * 2)),
                                          rng=rng, verbose=verbose,
                                          cell=cell if use_periodic else None)

    fit = fit_first_peak(nn, bin_width=bin_width)
    d_nn_median = float(np.median(nn))

    if shells:
        d_nn = float(shells[0])
        d_nn_source = 'rdf_first_shell'
    elif fit is not None:
        d_nn = float(fit['mu'])
        d_nn_source = 'nn_histogram_gaussian_fit'
        if verbose:
            print("  RDF gave no shells; using Gaussian fit of the NN histogram.")
    else:
        d_nn = d_nn_median
        d_nn_source = 'nn_median'
        if verbose:
            print("  RDF and peak fit both failed; using median NN distance.")

    sigma = float(fit['sigma']) if fit is not None else float(np.std(nn))

    detected, scores, ambiguous = classify_structure_from_shells(shells)

    dom = (dominant_structure or '').upper()
    hint_usable = dom in STRUCTURE_SHELL_RATIOS

    if hint_usable:
        chosen = dom
        structure_source = 'ptm_hint'
    elif detected is not None:
        chosen = detected
        structure_source = 'rdf_shell_ratios'
    else:
        chosen = 'UNKNOWN'
        structure_source = 'unidentified'

    chosen_err = scores.get(chosen)
    a_conv = (float(d_nn * NN_TO_CONVENTIONAL_A[chosen])
              if chosen in NN_TO_CONVENTIONAL_A else None)

    agrees = (detected == chosen) if (detected and chosen != 'UNKNOWN') else None

    if chosen == 'UNKNOWN' or chosen_err is None:
        reliability_score = 0.0
    else:
        reliability_score = float(max(0.0, 1.0 - chosen_err / STRUCTURE_ACCEPT_ERR))
        if ambiguous:
            reliability_score *= 0.6
        if agrees is False:
            reliability_score *= 0.5
    if chosen == 'HCP' and a_conv:
        c_val, c_source = hcp_c_from_shells(a_conv, shells)
        result = {
            'a': a_conv, 'b': a_conv, 'c': c_val,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'c_over_a': (float(c_val / a_conv) if c_val else None),
            'c_source': c_source,
        }
    elif a_conv:
        result = {
            'a': a_conv, 'b': a_conv, 'c': a_conv,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
        }
    else:
        result = {'nearest_neighbor': d_nn}

    result.update({
        'structure': chosen,
        'structure_source': structure_source,
        'cell_type': 'conventional' if a_conv else 'none',
        'rdf_detected_structure': detected,
        'rdf_agrees_with_structure': agrees,
        'ambiguous': bool(ambiguous),
        'shell_distances': [float(x) for x in shells],
        'shell_ratios': ([float(x / shells[0]) for x in shells] if shells else []),
        'shell_ratio_error': (float(chosen_err) if chosen_err is not None else None),
        'd_nn': float(d_nn),
        'd_nn_source': d_nn_source,
        'd_nn_median': d_nn_median,
        'd_nn_fit': (float(fit['mu']) if fit is not None else None),
        'd_nn_sigma': float(sigma),
        'reliability_score': reliability_score,
        'scores': {kk: float(vv) for kk, vv in scores.items()},
        'num_nn_samples': int(len(nn)),
    })

    if verbose:
        if chosen == 'UNKNOWN':
            print("  Structure NOT identified from shell ratios "
                  "(best residual exceeds the acceptance threshold).")
            print("  Reporting nearest-neighbour distance only: "
                  "d_nn = {:.4f} A".format(d_nn))
        else:
            print("  Structure: {} (from {}){}".format(
                chosen, structure_source, "  [ambiguous]" if ambiguous else ""))
            print("  d_nn = {:.4f} A ({}), a = {:.4f} A".format(
                d_nn, d_nn_source, result['a']))
            if agrees is False:
                print("  WARNING: RDF shell ratios favour {} (residual {:.4f}) "
                      "over the supplied hint {} (residual {})".format(
                          detected, scores.get(detected, float('nan')), chosen,
                          "n/a" if chosen_err is None else "%.4f" % chosen_err))
            print("  reliability_score = {:.3f}".format(reliability_score))

    diagnostics = {'nn_array': nn, 'fit': fit, 'used_periodic': use_periodic,
                   'rdf': rdf_diag, 'shells': shells}
    return result, diagnostics

def detect_structure_with_ovito(filepath, rmsd_cutoff=0.1, params=None):
    if not OVITO_AVAILABLE:
        return None, None, None, "OVITO not installed"

    try:
        pipeline = import_file(filepath)
        data = pipeline.compute()

        has_pbc = data.cell.pbc if hasattr(data.cell, 'pbc') else (False, False, False)
        is_periodic = any(has_pbc)

        ptm_modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff)
        pipeline.modifiers.append(ptm_modifier)
        data_with_ptm = pipeline.compute()

        if 'Structure Type' not in data_with_ptm.particles.keys():
            return None, None, None, "PTM analysis failed"

        struct_types = data_with_ptm.particles['Structure Type'].array
        unique_types, counts = np.unique(struct_types, return_counts=True)
        total_atoms = len(struct_types)

        type_map = {
            PolyhedralTemplateMatchingModifier.Type.OTHER.value: 'Other/Amorphous',
            PolyhedralTemplateMatchingModifier.Type.FCC.value: 'FCC',
            PolyhedralTemplateMatchingModifier.Type.HCP.value: 'HCP',
            PolyhedralTemplateMatchingModifier.Type.BCC.value: 'BCC',
            PolyhedralTemplateMatchingModifier.Type.ICO.value: 'ICO',
            PolyhedralTemplateMatchingModifier.Type.SC.value: 'SC',
            PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND.value: 'Cubic diamond',
            PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND.value: 'Hexagonal diamond',
            PolyhedralTemplateMatchingModifier.Type.GRAPHENE.value: 'Graphene',
        }

        dominant_structure = None
        max_count = 0
        structure_counts = {}

        for type_id, count in zip(unique_types, counts):
            name = type_map.get(type_id, f'Unknown_{type_id}')
            structure_counts[name] = count
            if name not in ['Other/Amorphous', 'Unknown'] and count > max_count:
                max_count = count
                dominant_structure = name

        lattice_info = None
        if is_periodic:
            cell_vectors = np.asarray(data.cell.matrix, dtype=float)[:, :3].T
            av, bv, cv = cell_vectors[0], cell_vectors[1], cell_vectors[2]

            def _ang(u, v):
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
                return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

            lattice_info = {
                'a': float(np.linalg.norm(av)),
                'b': float(np.linalg.norm(bv)),
                'c': float(np.linalg.norm(cv)),
                'alpha': _ang(bv, cv),
                'beta': _ang(av, cv),
                'gamma': _ang(av, bv),
                'cell_vectors': cell_vectors.tolist(),
                'pbc': has_pbc
            }

        return dominant_structure, structure_counts, lattice_info, None

    except Exception as e:
        return None, None, None, str(e)

def calculate_lattice_from_cluster(positions, structure_type, params=None,
                                   verbose=False, random_seed=0, cell=None):
    positions = np.asarray(positions, dtype=float)
    rng = np.random.RandomState(int(random_seed))

    k = params.neighbor_k if (params is not None and hasattr(params, 'neighbor_k')) else 20
    sample_size = (params.ls_sample_size
                   if (params is not None and hasattr(params, 'ls_sample_size'))
                   else 5000)

    neighbor_positions, _n_img = _periodic_images(positions, cell)
    tree = cKDTree(neighbor_positions)
    n_probe = min(int(sample_size), len(positions))
    probe_idx = rng.choice(len(positions), n_probe, replace=False)
    distances, _ = tree.query(positions[probe_idx],
                              k=min(int(k) + 1, len(neighbor_positions)))
    first_neighbor_dists = np.atleast_2d(distances)[:, 1]

    Q1, Q3 = np.percentile(first_neighbor_dists, [25, 75])
    IQR = Q3 - Q1
    filtered = first_neighbor_dists[
        (first_neighbor_dists >= Q1 - 1.5 * IQR) &
        (first_neighbor_dists <= Q3 + 1.5 * IQR)]
    if len(filtered) == 0:
        filtered = first_neighbor_dists
    d_nn_median = float(np.median(filtered))

    shells, _rdf = compute_rdf_shells(positions, sample_size=max(2000, n_probe),
                                      rng=rng, verbose=verbose, cell=cell)
    if shells:
        d_nn = float(shells[0])
        d_nn_source = 'rdf_first_shell'
    else:
        d_nn = d_nn_median
        d_nn_source = 'nn_median_fallback'

    detected, scores, ambiguous = classify_structure_from_shells(shells)

    if verbose:
        print("  Nearest-neighbour statistics:")
        print("    median (IQR-filtered, diagnostic): {:.4f} A".format(d_nn_median))
        print("    RDF first shell (used):            {:.4f} A".format(d_nn))
        if scores:
            print("    shell-ratio residuals: " +
                  ", ".join("%s=%.4f" % kv for kv in sorted(scores.items())))

    structure_type = structure_type.upper() if structure_type else "UNKNOWN"

    agrees = (detected == structure_type) if detected else None
    if verbose and agrees is False:
        print("    WARNING: RDF shell ratios favour {} over the supplied {}"
              .format(detected, structure_type))

    common = {
        'structure': structure_type,
        'method': 'rdf_first_shell',
        'd_nn': d_nn,
        'd_nn_source': d_nn_source,
        'd_nn_median': d_nn_median,
        'shell_distances': [float(x) for x in shells],
        'shell_ratios': ([float(x / shells[0]) for x in shells] if shells else []),
        'rdf_detected_structure': detected,
        'rdf_agrees_with_structure': agrees,
        'ambiguous': bool(ambiguous),
        'num_nn_samples': int(len(first_neighbor_dists)),
    }

    if structure_type in ('FCC', 'BCC'):
        a = float(d_nn * NN_TO_CONVENTIONAL_A[structure_type])
        lattice_info = dict(common)
        lattice_info.update({
            'a': a, 'b': a, 'c': a,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
            'cell_type': 'conventional_cubic',
            'volume': a ** 3,
        })

    elif structure_type == 'HCP':
        a = float(d_nn)
        c, c_source = hcp_c_from_shells(a, shells)
        lattice_info = dict(common)
        lattice_info.update({
            'a': a, 'b': a, 'c': float(c),
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'c_over_a': float(c / a),
            'c_source': c_source,
            'cell_type': 'conventional_hexagonal',
            'volume': float(a * a * c * math.sin(math.radians(120.0))),
        })
        if c_source == 'assumed_ideal':
            lattice_info['note'] = ('c/a NOT measured: the c-axis shell '
                                    '(multiplicity 2) was not resolved in the '
                                    'RDF; ideal sqrt(8/3) assumed')
            if verbose:
                print("    NOTE: c/a was not measured; ideal 1.633 assumed.")

    else:
        lattice_info = dict(common)
        lattice_info['nearest_neighbor'] = d_nn
        lattice_info['method'] = 'rdf_first_shell_nn_only'

    return lattice_info


def parse_extended_xyz_lattice(comment):

    if not comment:
        return None
    low = comment.lower()
    key = 'lattice='
    i = low.find(key)
    if i < 0:
        return None
    rest = comment[i + len(key):].lstrip()
    if not rest or rest[0] not in '"\'':
        return None
    quote = rest[0]
    j = rest.find(quote, 1)
    if j < 0:
        return None
    try:
        vals = [float(x) for x in rest[1:j].replace(',', ' ').split()]
    except ValueError:
        return None
    if len(vals) != 9:
        return None
    cell = np.asarray(vals, dtype=float).reshape(3, 3)
    return cell if _is_valid_cell(cell) else None


def read_xyz_file(filepath):

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        comment = lines[1] if len(lines) > 1 else ''
        cell = parse_extended_xyz_lattice(comment)

        positions = []
        elements = []
        for i in range(2, min(2 + num_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                elements.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
            elif len(parts) == 3:
                # No species column
                elements.append('X')
                positions.append([float(x) for x in parts[0:3]])

        if not positions:
            return None, None, None, None
        return np.array(positions), elements, len(positions), cell

    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        return None, None, None, None


def _positions_from_ovito(filepath):
    if not OVITO_AVAILABLE:
        return None, None
    try:
        data = import_file(filepath).compute()
        positions = np.asarray(data.particles.position.array, dtype=float)
        cell = None
        if getattr(data, 'cell', None) is not None:
            m = np.asarray(data.cell.matrix, dtype=float)
            if m.shape == (3, 4):
                m = m[:, :3]
            cand = np.asarray(m, dtype=float).T
            if _is_valid_cell(cand):
                cell = cand
        return positions, cell
    except Exception as e:
        print(f"  Could not read positions via OVITO: {e}")
        return None, None


def _reliability_label(result):
    if not isinstance(result, dict):
        return 'low'
    score = result.get('reliability_score')
    if score is None:
        return 'low'
    if result.get('structure') in (None, 'UNKNOWN'):
        return 'low'
    if score > 0.7:
        return 'high'
    if score > 0.3:
        return 'medium'
    return 'low'


def _classify_without_ptm(positions, verbose=False, random_seed=0,
                          sample_size=4000, cell=None):
    positions = np.asarray(positions, dtype=float)
    rng = np.random.RandomState(int(random_seed))

    neighbor_positions, _n_img = _periodic_images(positions, cell)
    tree = cKDTree(neighbor_positions)

    shells, _diag = compute_rdf_shells(positions, sample_size=sample_size,
                                       rng=rng, verbose=verbose, cell=cell)
    if shells:
        d_nn = float(shells[0])
    else:
        probe = positions[rng.choice(len(positions),
                                     min(2000, len(positions)), replace=False)]
        dd, _ = tree.query(probe, k=2)
        d_nn = float(np.median(dd[:, 1]))

    n_probe = min(int(sample_size), len(positions))
    probe_idx = rng.choice(len(positions), n_probe, replace=False)
    counts = np.asarray(tree.query_ball_point(positions[probe_idx], 1.2 * d_nn,
                                              return_length=True)) - 1
    counts = counts[counts >= 0]
    cn_mode = int(np.bincount(counts).argmax()) if counts.size else 0

    detected, scores, ambiguous = classify_structure_from_shells(shells)

    if detected is not None:
        structure = detected
    elif cn_mode >= 13:
        structure = 'BCC'
    elif 10 <= cn_mode <= 12:
        structure = 'FCC'    
    else:
        structure = 'UNKNOWN'

    if verbose:
        print(f"  modal coordination number (r < 1.2 d_nn) = {cn_mode}")
        if scores:
            print("  shell-ratio residuals: " +
                  ", ".join("%s=%.4f" % kv for kv in sorted(scores.items())))
        if structure in ('FCC', 'HCP'):
            print("  NOTE: FCC and HCP are not separable from shell ratios; "
                  "run stage 1 (PTM) to distinguish them.")

    return structure, cn_mode, shells


def analyze_structure_robust(filepath, params):

    verbose = params.verbose

    print("=" * 60)
    print(f"Analyzing structure: {os.path.basename(filepath)}")
    print("=" * 60)

    all_results = []  
    vector_lengths_data = None
    pairwise_distances_data = None

    if OVITO_AVAILABLE:
        if verbose:
            print("\n[Method 1] Using structure analysis...")

        dominant_structure, structure_counts, lattice_info, error = detect_structure_with_ovito(
            filepath, params.rmsd_cutoff
        )

        if error:
            if verbose:
                print(f"  OVITO analysis warning: {error}")
        else:
            if verbose:
                print(f"  Detected structure: {dominant_structure}")
                print(f"  Structure distribution: {structure_counts}")

            pbc_flags = tuple(lattice_info['pbc']) if lattice_info else (False, False, False)
            if lattice_info and any(pbc_flags):
                print("\n  PERIODIC SYSTEM DETECTED  (pbc = %s)" % (pbc_flags,))
                print("  Simulation cell (SUPERCELL, not the unit cell):")
                print(f"    A = {lattice_info['a']:.4f} A")
                print(f"    B = {lattice_info['b']:.4f} A")
                print(f"    C = {lattice_info['c']:.4f} A")
                print(f"    alpha = {lattice_info['alpha']:.2f} deg, "
                      f"beta = {lattice_info['beta']:.2f} deg, "
                      f"gamma = {lattice_info['gamma']:.2f} deg")
                if not all(pbc_flags):
                    print("    NOTE: mixed boundary conditions; the non-periodic "
                          "direction(s) contain free surfaces.")

                supercell = {
                    'supercell_A': float(lattice_info['a']),
                    'supercell_B': float(lattice_info['b']),
                    'supercell_C': float(lattice_info['c']),
                    'supercell_alpha': float(lattice_info['alpha']),
                    'supercell_beta': float(lattice_info['beta']),
                    'supercell_gamma': float(lattice_info['gamma']),
                    'pbc': pbc_flags,
                }

                positions, elements, num_atoms, cell = read_xyz_file(filepath)
                if positions is None:
                    positions, cell = _positions_from_ovito(filepath)
                if positions is None:
                    print("  ERROR: Could not read atomic positions.")
                    return None

                lattice_est = calculate_lattice_from_cluster(
                    positions, dominant_structure, params=params, verbose=verbose,
                    random_seed=params.vector_stats_random_seed, cell=cell)
                lattice_est.update(supercell)

                if lattice_est.get('a'):
                    reps = [lattice_info[key] / lattice_est['a'] for key in ('a', 'b', 'c')]
                    lattice_est['supercell_repeats'] = [float(x) for x in reps]
                    print("  Measured lattice constant a = {:.4f} A".format(lattice_est['a']))
                    print("  Supercell / a = {}  (should be near-integer for a "
                          "commensurate box)".format([round(x, 2) for x in reps]))
                    max_dev = max(abs(x - round(x)) for x in reps)
                    lattice_est['supercell_commensurate'] = bool(max_dev < 0.05)
                    if max_dev >= 0.05:
                        print("  WARNING: box/a is not close to integer "
                              "(max deviation {:.3f}); the cell may be strained, "
                              "non-cubic, or the structure misidentified.".format(max_dev))

                all_results.append({
                    'structure': dominant_structure,
                    'lattice': lattice_est,
                    'method': 'rdf_first_shell_pbc',
                    'reliability': ('high' if lattice_est.get('supercell_commensurate')
                                    else 'medium'),
                })

                if params.use_advanced_methods:
                    advanced_result, vec_stats_data = extract_lattice_by_vector_statistics_improved(
                        positions, dominant_structure, params, verbose=verbose,
                        random_seed=params.vector_stats_random_seed
                    )
                    if advanced_result:
                        advanced_result.update(supercell)
                        all_results.append({
                            'structure': dominant_structure,
                            'lattice': advanced_result,
                            'method': 'vector_stats_pbc',
                            'reliability': 'high'
                        })
                        if vec_stats_data and 'all_vector_lengths' in vec_stats_data:
                            vector_lengths_data = vec_stats_data['all_vector_lengths']
                        elif vec_stats_data and 'lengths' in vec_stats_data:
                            vector_lengths_data = vec_stats_data['lengths']

                    ls_result, ls_diagnostics = extract_lattice_auto(
                        positions, dominant_structure, cell=cell,
                        periodic=params.ls_use_periodic,
                        sample_size=params.ls_sample_size,
                        bin_width=params.ls_bin_width,
                        k=params.ls_k_neighbors,
                        verbose=verbose,
                        random_seed=params.vector_stats_random_seed
                    )
                    if ls_result:
                        all_results.append({
                            'structure': ls_result.get('structure', dominant_structure),
                            'lattice': ls_result,
                            'method': 'rdf_shell_ratios',
                            'reliability': _reliability_label(ls_result)
                        })
                        if ls_diagnostics and 'nn_array' in ls_diagnostics:
                            pairwise_distances_data = ls_diagnostics['nn_array']

                    _plot_comparison_if_requested(params, vector_lengths_data,
                                                  pairwise_distances_data)

                return all_results
            else:
                print("\n  NON-PERIODIC SYSTEM (CLUSTER/SURFACE)")
                print(f"  Structure identified as: {dominant_structure}")

                positions, elements, num_atoms, cell = read_xyz_file(filepath)
                if positions is None:
                    positions, cell = _positions_from_ovito(filepath)
                if positions is None:
                    print("  ERROR: Could not read atomic positions.")
                    return None

                print("\n  [Method 1] Nearest-neighbour distance from the RDF...")
                lattice_est = calculate_lattice_from_cluster(
                    positions, dominant_structure, params=params, verbose=verbose,
                    random_seed=params.vector_stats_random_seed
                )


                if 'a' in lattice_est:
                    print(f"\n  Original method estimated lattice parameters:")
                    print(f"    a = {lattice_est.get('a', 0):.4f} Å, " +
                          f"b = {lattice_est.get('b', lattice_est.get('a', 0)):.4f} Å, " +
                          f"c = {lattice_est.get('c', lattice_est.get('a', 0)):.4f} Å")
                    if 'alpha' in lattice_est:
                        print(f"    α = {lattice_est['alpha']:.2f}°, " +
                              f"β = {lattice_est['beta']:.2f}°, " +
                              f"γ = {lattice_est['gamma']:.2f}°")

                result = {
                    'structure': dominant_structure,
                    'lattice': lattice_est,
                    'method': 'rdf_first_shell',
                    'reliability': 'medium'
                }
                all_results.append(result)

                if params.use_advanced_methods and positions is not None:
                    print("\n" + "-" * 60)
                    print("ADVANCED ANALYSIS FOR POLYCRYSTALS")
                    print("-" * 60)


                    vec_stats_result, vec_stats_data = extract_lattice_by_vector_statistics_improved(
                        positions, dominant_structure, params, verbose=verbose,
                        random_seed=params.vector_stats_random_seed
                    )

                    if vec_stats_result:

                        reliability = ('high' if vec_stats_result.get('num_clusters', 0) >= 6
                                       else 'medium')

                        result2 = {
                            'structure': dominant_structure,
                            'lattice': vec_stats_result,
                            'method': 'vector_statistics',
                            'reliability': reliability
                        }
                        all_results.append(result2)
                        if vec_stats_data and 'all_vector_lengths' in vec_stats_data:
                            vector_lengths_data = vec_stats_data['all_vector_lengths']
                        elif vec_stats_data and 'lengths' in vec_stats_data:
                            vector_lengths_data = vec_stats_data['lengths']

                    ls_result, ls_diagnostics = extract_lattice_auto(
                        positions, dominant_structure, cell=cell,
                        periodic=params.ls_use_periodic,
                        sample_size=params.ls_sample_size,
                        bin_width=params.ls_bin_width,
                        k=params.ls_k_neighbors,
                        verbose=verbose,
                        random_seed=params.vector_stats_random_seed
                    )

                    if ls_result:
                        result3 = {
                            'structure': ls_result.get('structure', dominant_structure),
                            'lattice': ls_result,
                            'method': 'rdf_shell_ratios',
                            'reliability': _reliability_label(ls_result)
                        }
                        all_results.append(result3)
                        if ls_diagnostics and 'nn_array' in ls_diagnostics:
                            pairwise_distances_data = ls_diagnostics['nn_array']

                    _plot_comparison_if_requested(params, vector_lengths_data,
                                                  pairwise_distances_data)


                    if len(all_results) > 1 and verbose:
                        print("\n" + "-" * 60)
                        print("COMPARISON OF DIFFERENT METHODS")
                        print("-" * 60)

                        for i, res in enumerate(all_results):
                            method = res['method']
                            lat = res['lattice']
                            if 'a' in lat:
                                print(f"Method {i+1} ({method}):")
                                print(f"  a={lat.get('a', 0):.4f}, b={lat.get('b', 0):.4f}, c={lat.get('c', 0):.4f}")
                                if 'alpha' in lat:
                                    print(f"  α={lat['alpha']:.2f}°, β={lat['beta']:.2f}°, γ={lat['gamma']:.2f}°")
                                print(f"  Reliability: {res['reliability']}")

    if verbose and not all_results:
        print("\n[Method 2] Fallback analysis (no OVITO)...")

    if not all_results:
        positions, elements, num_atoms, cell = read_xyz_file(filepath)
        if positions is None:
            positions, cell = _positions_from_ovito(filepath)
            num_atoms = len(positions) if positions is not None else None
        if positions is None:
            print("  ERROR: Could not read file")
            return None

        print(f"  Atoms loaded: {num_atoms}")

        if len(positions) > 10:
            guessed_structure, cn_mode, shells_fb = _classify_without_ptm(
                positions, verbose=verbose,
                random_seed=params.vector_stats_random_seed, cell=cell)

            print(f"  Structure from coordination + shell ratios: {guessed_structure}"
                  f"  (modal coordination number {cn_mode})")

            lattice_est = calculate_lattice_from_cluster(
                positions, guessed_structure, params=params, verbose=verbose,
                random_seed=params.vector_stats_random_seed, cell=cell)

            print(f"\n  WARNING: PTM unavailable; structure assignment is less reliable")
            print(f"  Estimated parameters (use with caution):")
            for key, value in lattice_est.items():
                if key not in ['structure', 'method', 'note']:
                    if isinstance(value, float):
                        print(f"    {key} = {value:.4f}")
                    else:
                        print(f"    {key} = {value}")

            result = {
                'structure': guessed_structure,
                'lattice': lattice_est,
                'method': 'fallback_estimation',
                'reliability': 'low'
            }
            all_results.append(result)

    return all_results if all_results else None

def _float_or_auto(value):
    if str(value).strip().lower() == 'auto':
        return 'auto'
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected a distance in A or 'auto', got %r" % value)

def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith('.txt'):
        param_file = sys.argv[1]
        param_dict = read_parameter_file(param_file)

        if param_dict is None:
            print(f"Error: Could not read parameter file '{param_file}'")
            sys.exit(1)

        if 'input' not in param_dict:
            print("Error: 'input' parameter not found in parameter file")
            sys.exit(1)

        input_file = param_dict['input']

        params = AnalysisParameters.from_dict(param_dict)

    else:

        _d = AnalysisParameters()

        parser = argparse.ArgumentParser(
            description="Lattice parameter analysis for periodic and non-periodic systems."
        )

        parser.add_argument('--input', type=str, required=True,
                           help='Path to input structure file (XYZ, LAMMPS dump, etc.)')

        parser.add_argument('--rmsd_cutoff', type=float, default=_d.rmsd_cutoff,
                           help='RMSD cutoff for PTM analysis (default: %(default)s)')

        parser.add_argument('--verbose', dest='verbose', action='store_true',
                           default=_d.verbose,
                           help='Print detailed analysis information (default: on)')
        parser.add_argument('--quiet', dest='verbose', action='store_false',
                           help='Suppress detailed analysis information')

        parser.add_argument('--no_advanced', action='store_true',
                           help='Disable advanced lattice extraction methods')

        parser.add_argument('--plot', action='store_true',
                           help='Generate plots during analysis')

        parser.add_argument('--neighbor_k', '--neighbour_k', type=int,
                           dest='neighbor_k', default=_d.neighbor_k,
                           help='Neighbours per atom for NN statistics (default: %(default)s)')

        parser.add_argument('--vector_stats_random_seed', type=int,
                           default=_d.vector_stats_random_seed,
                           help='Random seed for reproducibility (default: %(default)s)')

        parser.add_argument('--vector_stats_min_neighbors', '--vector_stats_min_neighbours',
                           type=int, dest='vector_stats_min_neighbors',
                           default=_d.vector_stats_min_neighbors,
                           help='DBSCAN min_samples for direction clusters (default: %(default)s)')

        parser.add_argument('--vector_stats_k_nn', type=int, default=_d.vector_stats_k_nn,
                           help='Neighbours per sampled atom for the vector search '
                                '(default: %(default)s; ~14 for BCC, ~18 for FCC)')

        parser.add_argument('--vector_stats_max_cutoff', type=_float_or_auto,
                           default=_d.vector_stats_max_cutoff,
                           help="Maximum neighbour distance in A, or 'auto' for "
                                "1.45 x d_nn (default: %(default)s)")

        parser.add_argument('--vector_stats_dbscan_eps', type=float,
                           default=_d.vector_stats_dbscan_eps,
                           help='Direction-cluster tolerance, chord distance on the '
                                'unit sphere (default: %(default)s ~ 4 deg)')

        parser.add_argument('--vector_stats_merge_threshold', type=float,
                           default=_d.vector_stats_merge_threshold,
                           help='|cos| above which two direction clusters are one '
                                'lattice axis (default: %(default)s)')

        parser.add_argument('--vector_stats_tolerance_factor', type=float,
                           default=_d.vector_stats_tolerance_factor,
                           help='Relative width of the first-shell vector filter '
                                '(default: %(default)s)')

        parser.add_argument('--vector_stats_sample_size', type=int,
                           default=_d.vector_stats_sample_size,
                           help='Atoms sampled for vector statistics (default: %(default)s)')

        parser.add_argument('--vector_stats_hist_bins', type=int,
                           default=_d.vector_stats_hist_bins,
                           help='Histogram bins for diagnostics (default: %(default)s)')

        parser.add_argument('--ls_sample_size', type=int, default=_d.ls_sample_size,
                           help='Atoms sampled for NN/RDF statistics (default: %(default)s)')

        parser.add_argument('--ls_bin_width', type=float, default=_d.ls_bin_width,
                           help='Bin width in A for the NN histogram fit (default: %(default)s)')

        parser.add_argument('--ls_k_neighbors', '--ls_k_neighbours', type=int,
                           dest='ls_k_neighbors', default=_d.ls_k_neighbors,
                           help='Neighbours per atom for the KD-tree query (default: %(default)s)')

        parser.add_argument('--ls_use_periodic', type=str, default=None,
                           help='Force periodic mode (true/false); omit for auto-detect')


        args = parser.parse_args()

        input_file = args.input

        params = AnalysisParameters()
        params.rmsd_cutoff = args.rmsd_cutoff
        params.verbose = args.verbose
        params.use_advanced_methods = not args.no_advanced
        params.plot = args.plot
        params.neighbor_k = args.neighbor_k

        params.vector_stats_random_seed = args.vector_stats_random_seed
        params.vector_stats_min_neighbors = args.vector_stats_min_neighbors
        params.vector_stats_k_nn = args.vector_stats_k_nn
        params.vector_stats_max_cutoff = args.vector_stats_max_cutoff
        params.vector_stats_dbscan_eps = args.vector_stats_dbscan_eps
        params.vector_stats_merge_threshold = args.vector_stats_merge_threshold
        params.vector_stats_tolerance_factor = args.vector_stats_tolerance_factor
        params.vector_stats_sample_size = args.vector_stats_sample_size
        params.vector_stats_hist_bins = args.vector_stats_hist_bins

        params.ls_sample_size = args.ls_sample_size
        params.ls_bin_width = args.ls_bin_width
        params.ls_k_neighbors = args.ls_k_neighbors

        if args.ls_use_periodic is not None:
            if args.ls_use_periodic.lower() in ['true', 'yes', '1']:
                params.ls_use_periodic = True
            elif args.ls_use_periodic.lower() in ['false', 'no', '0']:
                params.ls_use_periodic = False
            else:
                params.ls_use_periodic = None 

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    results = analyze_structure_robust(input_file, params)

    if results:
        output_file = os.path.splitext(input_file)[0] + "_lattice_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"COMPREHENSIVE LATTICE ANALYSIS RESULTS\n")
            f.write(f"File: {os.path.basename(input_file)}\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results):
                f.write(f"\n{'='*40}\n")
                f.write(f"METHOD {i+1}: {result['method'].upper()}\n")
                f.write(f"{'='*40}\n")
                f.write(f"Primary crystal structure: {result['structure']}\n")
                f.write(f"Reliability: {result['reliability']}\n\n")

                lattice = result['lattice']
                f.write("Lattice Parameters:\n")

                if 'a' in lattice:
                    f.write(f"  a = {lattice.get('a', 0):.4f} Å\n")
                    f.write(f"  b = {lattice.get('b', lattice.get('a', 0)):.4f} Å\n")
                    f.write(f"  c = {lattice.get('c', lattice.get('a', 0)):.4f} Å\n")

                if 'alpha' in lattice:
                    f.write(f"  α = {lattice['alpha']:.2f}°\n")
                    f.write(f"  β = {lattice['beta']:.2f}°\n")
                    f.write(f"  γ = {lattice['gamma']:.2f}°\n")

                for key, value in lattice.items():
                    if key not in ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'structure', 'method', 'pbc']:
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        elif isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], list):
                                f.write(f"  {key}: [list of {len(value)} vectors]\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                        else:
                            f.write(f"  {key}: {value}\n")

                f.write("\n")

            f.write(f"\n{'='*80}\n")
            f.write("SUMMARY AND RECOMMENDATIONS\n")
            f.write(f"{'='*80}\n\n")

            reliability_order = {'high': 3, 'medium': 2, 'low': 1}
            best_result = max(results, key=lambda x: reliability_order.get(x['reliability'], 0))

            f.write(f"Recommended result (from {best_result['method']}, reliability: {best_result['reliability']}):\n")
            lattice = best_result['lattice']
            if 'a' in lattice:
                f.write(f"  Crystal system: {best_result['structure']}\n")
                f.write(f"  Lattice constants: a={lattice.get('a', 0):.4f} Å, " +
                       f"b={lattice.get('b', lattice.get('a', 0)):.4f} Å, " +
                       f"c={lattice.get('c', lattice.get('a', 0)):.4f} Å\n")
                if 'alpha' in lattice:
                    f.write(f"  Lattice angles: α={lattice['alpha']:.2f}°, " +
                           f"β={lattice['beta']:.2f}°, γ={lattice['gamma']:.2f}°\n")

            f.write(f"\nTotal methods compared: {len(results)}\n")
            f.write("Note: For polycrystalline samples, the vector statistics method is most robust to grain rotations.\n")

        print(f"\n" + "=" * 60)
        print(f"Analysis complete! Results saved to: {output_file}")
        print("=" * 60)

        if params.verbose:
            print("\nSUMMARY OF ALL METHODS:")
            for i, result in enumerate(results):
                method = result['method']
                reliability = result['reliability']
                lattice = result['lattice']
                if 'a' in lattice:
                    print(f"{i+1}. {method} ({reliability}): a={lattice.get('a', 0):.4f} Å, " +
                          f"α={lattice.get('alpha', 0):.2f}°")

    if not OVITO_AVAILABLE:
        print("\n" + "!" * 60)
        print("WARNING: OVITO Python module is not installed!")
        print("For state-of-the-art structure detection, install OVITO:")
        print("  conda install -c ovito ovito")
        print("Or visit: https://www.ovito.org/download/")
        print("!" * 60)

if __name__ == '__main__':
    main()