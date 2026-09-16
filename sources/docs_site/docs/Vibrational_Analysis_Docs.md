# Vibrational Analysis Module

## Overview
This module performs advanced vibrational analysis using molecular dynamics trajectory data. It consists of two workflows:

1.  **Vibrational Density of States (VDOS)**: Computed from the power spectrum of the atomic velocities (equivalently, the Fourier transform of the Velocity Autocorrelation Function, VACF).
2.  **Infrared (IR) Spectroscopy**: Computed from the power spectrum of the dipole current $\dot{\mathbf{M}}(t)$ (equivalently, the Fourier transform of the Dipole-Derivative Autocorrelation Function, DACF).

Both workflows share one signal-processing core, offering **two independent, mutually validating spectral estimators**, a decaying lag window, and optional quantum correction factors.

---

## Part I: Vibrational Density of States (VDOS)

### Physics and Methodology
The VDOS, $g(\omega)$, represents the distribution of vibrational modes available to the system. It is calculated as the **mass-weighted** velocity power spectrum:

$$ g(\nu) = \sum_i m_i \, S_i(\nu), \qquad S_i(\nu) = \text{PSD}\!\left[\mathbf{v}_i(t)\right] $$

where $S_i$ is the one-sided power spectral density of atom $i$'s velocity **vector**, summed over $x$, $y$ and $z$ with their true relative weights. Equivalently, via the Wiener–Khinchin theorem,

$$ g(\nu) \propto \sum_i m_i \int_{-\infty}^{\infty} e^{-i\omega t} \langle \mathbf{v}_i(t) \cdot \mathbf{v}_i(0) \rangle \, dt $$

The mass weighting is applied **explicitly**. There is no per-atom or per-component normalisation of the autocorrelation function, so the velocity sum rule below remains available as a validation of the entire pipeline.

The code supports two modes:

*   **Full Mode**: Sums the mass-weighted velocity power spectrum over all atoms.
*   **Bond Mode**: Differentiates the scalar bond length of a specified atom pair, which is equivalent to projecting their relative velocity onto the bond unit vector, isolating that bond's stretching modes. (Not mass weighted — it is a single internal coordinate.)

### Validation: the velocity sum rule
For a classical system in equilibrium, equipartition gives $\langle m_i v_{i\alpha}^2\rangle = k_BT$ for every Cartesian degree of freedom, hence

$$ \sum_i m_i \langle v_i^2 \rangle = N_{\text{dof}} \, k_B T $$

Integrating the mass-weighted PSD must recover the same quantity. `vdos.py` therefore prints an **effective temperature** $T_{\text{eff}}$ obtained from the spectral integral. This is the single most useful check on the whole calculation:

*   $T_{\text{eff}}$ should match your MD thermostat temperature.
*   A large mismatch points to a wrong `delta_t`, a wrong `nmeasure`, wrong trajectory or velocity units, an incorrect `temperature` parameter, unaccounted constraints, or poor equilibration.
*   With constraints (rigid bonds via SHAKE/RATTLE), subtract the constrained degrees of freedom from $N_{\text{dof}}$. Note that rigid X–H bonds remove the X–H stretch band entirely.
*   `N_dof` is reported as $3N-3$ when `Center_of_mass_correction = True`, else $3N$.

### Features (VDOS & IR)
*   **Box Inference**: If the input has no cell, each box length is estimated from the median jump of wrapped coordinates between frames, falling back to the coordinate extent when nothing wraps. No minimum box length is imposed.
*   **Center of Mass (COM) Correction**: Removes the net translational motion of the system to prevent spurious low-frequency peaks. (If not applied, especially for gas-phase molecules/clusters, a massive peak at/near 0 cm⁻¹ caused by translational motion masks true vibrations.)
*   **PBC Unwrapping**: Unwraps atomic positions across periodic boundaries to ensure continuous trajectories. Essential: a wrapped coordinate produces a delta-function velocity spike at the wrap frame, which spreads white noise across the entire spectrum. Note the unwrapping is per-Cartesian-direction and therefore assumes an **orthorhombic** cell. For NetCDF input the frame-averaged `cell_lengths` are used whenever positions are differentiated; the output header records whether unwrapping was applied.
*   **Velocity source**: for VDOS, velocities stored in a NetCDF trajectory are used unless `force_numerical = True`. Text trajectories (XYZ / LAMMPS dump), and all IR calculations, obtain velocities numerically as $\nabla$(positions). Central differences act as a low-pass filter $\mathrm{sinc}(\omega\Delta t)$ — negligible (1.5 % at 4500 cm⁻¹) for $\Delta t = 0.25$ fs, but 66 % at an effective $\Delta t$ of 2 fs. A warning is printed when the loss exceeds 5 %.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_file` | File Path | `traj.xyz` | Input trajectory (XYZ, LAMMPS dump, or NetCDF). ASE is used for text trajectories and content-sniffs LAMMPS dumps even when named `.xyz`. Text trajectories must use the `.xyz` extension: any other extension (e.g. `.lammpstrj`) is read as NetCDF. |
| `mode` | Enum | `full` | Analysis mode: `full` (all atoms) or `bond` (specific pair). |
| `bond_indices` | Integers | None | **(Bond Mode Only)** Indices of the two atoms defining the bond (1-based). |
| `delta_t` | Float | 0.25 | Time step between MD frames (femtoseconds). |
| `spectral_estimator` | Enum | `acf` | `acf` (Blackman–Tukey) or `welch` (averaged periodogram). See Part III. |
| `window_kind` | Enum | `Gaussian` | Window function type (see Windowing Functions). |
| `window_width_ps`| Float | 1.0 | **FWHM** of the Gaussian lag window, in picoseconds. Sets the spectral resolution (~29 cm⁻¹ at 1.0 ps). Only the Gaussian window uses it; Hann, Hamming and Blackman-Harris lag windows always span the full trajectory. It is also the default `welch_segment_ps`. |
| `welch_segment_ps` | Float | = `window_width_ps` | **(`welch` only)** Segment duration in picoseconds. |
| `welch_overlap` | Float | 0.5 | **(`welch` only)** Fractional segment overlap, in [0, 1). |
| `quantum_correction`| Boolean | **`False`** | See Quantum Correction. Off by default for VDOS. |
| `temperature` | Float | 300.0 | System temperature (Kelvin), used by the quantum correction and reported against $T_{\text{eff}}$. |
| `PBC` | Boolean | `False` | Toggles Periodic Boundary Condition handling/unwrapping. |
| `Center_of_mass_correction`| Boolean | `True` | Removes global translation from velocity data. |
| `masses` | List | `C 12.011; H 1.008` | Atomic masses, e.g. `C 12.011; H 1.008`. An element present in the trajectory but absent here is a **hard error** — a silent fallback mass would corrupt both the COM correction and the mass weighting. |
| `force_numerical`| Boolean | `False` | If `True`, recalculates velocities from positions even if velocities are present in the file. Only NetCDF files carry stored velocities. |
| `nskip` / `nmeasure` | Integer | 0 / 1 | Initial frames to skip / downsampling factor. |
| `plot_min_wavenumber` | Float | 50.0 | **Display only.** Lower plot limit (cm⁻¹). |
| `plot_max_wavenumber` | Float | 4500.0 | **Display only.** Upper plot limit (cm⁻¹). Warns if above Nyquist. |
| `symbols` | List | None | **(NetCDF only)** Chemical symbols, one per atom in trajectory order. Required, since NetCDF carries no element information. |
| `velocity_unit` | Enum | `angstrom/ps` | **(NetCDF only)** Units of *stored* velocities: `angstrom/ps` (AMBER convention), `angstrom/fs`, or `angstrom/s`. A wrong choice shows up immediately as an absurd $T_{\text{eff}}$. |
| `output_data` | File Path | `VDOS.txt` | Raw spectrum output (see Output metadata). |
| `output_plot` | File Path | `VDOS.png` | Plot output; the extension selects the image format. |
| `dpi` | Integer | `150` | Plot resolution. |

`use_normalized_vectors` is no longer supported: setting it to `True` aborts with an explanatory error, because neither $|\mathbf{r}|$ nor $|\mathbf{v}|$ yields a density of states.


---

## Part II: Infrared (IR) Spectra

### Physics and Methodology
The IR absorption spectrum is obtained from the autocorrelation of the **dipole current** (dipole derivative) $\dot{\mathbf{M}}(t) \equiv \mathbf{J}(t)$:

$$ I(\nu) \propto \int_{-\infty}^{\infty} e^{-i\omega t} \langle \dot{\mathbf{M}}(t) \cdot \dot{\mathbf{M}}(0) \rangle \, dt $$

$$ \mathbf{J}(t) = \sum_{i} q_i(t)\, \mathbf{v}_i(t) \; + \; \sum_{i} \dot{q}_i(t)\, \mathbf{r}_i(t) $$

Correlating the dipole **derivative** rather than the dipole itself is deliberate and important:

*   Velocities are independent of the periodic-image choice, whereas $\sum_i q_i \mathbf{r}_i$ is not. This is what makes the spectrum well defined under PBC.
*   It supplies the $\omega^2$ weighting that IR absorption requires, since $S_{\dot M}(\omega) = \omega^2 S_M(\omega)$.

The three Cartesian components of $\mathbf{J}$ are combined into **one** vector correlation / PSD with their true relative weights. They are never normalised independently — doing so forces $C_\alpha(0)=1$ for each and destroys both the polarisation weighting and the absolute intensity.

*   **Static Charge Mode**: constant charges, so $\dot q = 0$ and the second term vanishes. $\mathbf{J}(t) = \sum_i q_i \mathbf{v}_i(t)$.
*   **Dynamic Charge Mode**: charges read per frame from a LAMMPS dump `q` column; $\dot q$ obtained by a Savitzky–Golay derivative filter.

> **Limitation of static charges.** Fixed charges omit charge flux, which for C–H stretches is comparable in magnitude to the $q\mathbf{v}$ term. **Peak positions are reliable; relative IR intensities from a fixed-charge model are qualitative only.** Choose `static_charges` from the force field that produced the MD, not by hand.

> **Note on the dynamic $\dot q \mathbf{r}$ term.** It is origin- and periodic-image-dependent, so it is physically meaningful only for a non-periodic system or a single unwrapped molecule.

### Charge neutrality
`static_charges` are validated **after** assignment against `charge_tolerance`. A non-zero total charge contaminates $\mathbf{J} = \sum_i q_i \mathbf{v}_i$ with overall translation: the COM correction enforces $\sum_i m_i \mathbf{v}_i = 0$, **not** $\sum_i \mathbf{v}_i = 0$, so with unequal masses the net-charge term does not cancel. The composition, per-element subtotals and total charge are printed every run.

### Input Parameters
In addition to the shared parameters above (`spectral_estimator`, `window_kind`, `window_width_ps`, `welch_*`, `plot_*_wavenumber`, `delta_t`, `nskip`, `nmeasure`, `PBC`, `Center_of_mass_correction`, `masses`, `temperature`):

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `charge_mode` | Enum | `static` | `static` or `dynamic`. Dynamic mode needs a LAMMPS custom dump with a `q` column. |
| `static_charges` | Dictionary | None | Charges for static mode, e.g. `C -0.117; H 0.06` (neutral for the C₂₀H₃₉ example trajectory). Every element in the trajectory must appear, otherwise it is silently invisible to the dipole current. |
| `charge_tolerance` | Float | 1e-6 | Maximum allowed $\lvert\sum_i q_i\rvert$ in units of *e*. |
| `charge_tolerance_fatal` | Boolean | `False` | `True` aborts instead of warning when the tolerance is exceeded. |
| `quantum_correction`| Boolean | `True` | Applies the **harmonic** QCF exactly once. |
| `dqdt_savgol_window` | Integer | `5` | **(Dynamic mode only)** Savitzky–Golay window, in frames, for the $\dot q$ derivative. Must be odd and at least 5. |
| `ir_output` | File Path | `ir_spectrum.txt` | Raw spectrum output (see Output metadata). |
| `ir_plot` | File Path | `ir_spectrum.png` | Plot output file. |
| `dpi` | Integer | `300` | Plot resolution. |
| `dqdt_savgol_window` | Integer | 5 | **(dynamic only)** Savitzky–Golay window length (odd, ≥ 5) for $\dot q$. |
| `ir_output` / `ir_plot` | File Path | `ir_spectrum.txt` / `.png` | Output data and figure. |

---

## Part III: Signal Processing & Corrections

### Spectral estimators (`spectral_estimator`)
Both estimators return a **one-sided power spectral density on the same physical scale**, so they can be compared directly on the same trajectory and should agree to within their statistical error. Running both is a good cross-check.

#### `acf` — Blackman–Tukey (default)
1.  **Biased, mean-centred vector ACF**, summed over Cartesian components:

    $$ C(k) = \frac{1}{N}\sum_{\alpha}\sum_{n} x_\alpha(n)\,x_\alpha(n+k) $$

    The **biased** ($1/N$) divisor is used deliberately, not $1/(N-k)$. The biased estimator is positive semi-definite — its full-lag transform is exactly the periodogram — so the resulting spectrum cannot go negative. The unbiased $1/(N-k)$ estimator has no such property, has large variance at long lag (where only a handful of samples contribute), and can produce negative spectral density.

2.  **Decaying lag window** $w$ with $w(0)=1$, applied to the one-sided ACF.
3.  **Transform**:

    $$ S_1(\nu) = 2\,\Delta t \left( 2\,\mathrm{Re}\!\left[\mathrm{rFFT}(Cw)\right] - C(0)w(0) \right) $$

    with the DC and Nyquist bins un-doubled, and $n_{\text{fft}} \ge 2N-1$ to avoid circular wrap-around.

#### `welch` — averaged modified periodogram
The trajectory is split into overlapping segments; each is mean-centred (`detrend='constant'`), multiplied by a **centred** signal window, transformed, and the periodograms are averaged (Welch, *IEEE Trans. Audio Electroacoust.* **15**, 70 (1967)). The window is applied **per segment by `scipy.signal.welch`**; the series is never pre-multiplied by a window.

#### Choosing between them
At matched resolution the two have comparable statistical variance (Welch's advantage measures ~1.3× in variance, i.e. ~12 % in standard error). Practical differences:

| | `acf` | `welch` |
| :--- | :--- | :--- |
| Frequency grid | Fine (zero-padded; interpolation, not extra information) | Coarser, set by segment length |
| Low frequencies | Retains content down to $1/T_{\text{traj}}$ | Per-segment detrending suppresses content below $1/T_{\text{seg}}$ |
| Non-negativity | Guaranteed with a Gaussian lag window | Always guaranteed |
| Speed | Slower | Faster |

Both are reported with their **bin spacing** and their **effective resolution** (the measured FWHM of the spectral window). These are different numbers: zero padding makes the bin spacing finer without adding information.

### Windowing Functions (`window_kind`)
Finite truncation of the correlation function introduces spectral leakage. A window is applied to suppress it.

**Critically, the `acf` route requires a *lag* window, not a signal window**: a monotonically decaying function with $w(0)=1$, built here as the second half of the corresponding symmetric window. Every `scipy.signal.windows.*` function is *centre-peaked*; applying one directly to a one-sided ACF suppresses the zero-lag region — where most of the correlation information lives — while amplifying the noisy long-lag tail.

*   **Gaussian** (Default, recommended): configurable width (`window_width_ps`, interpreted as FWHM). It is the only supported window whose *spectral* window is non-negative, which guarantees $S(\nu)\ge 0$.
*   **Blackman-Harris**: strongest sidelobe suppression, at the cost of broader peaks.
*   **Hamming**: good general-purpose window.
*   **Hann**: raised cosine; good balance between main-lobe width and sidelobe suppression.

Hann, Hamming and Blackman-Harris lag windows have small negative spectral sidelobes and can in principle produce slightly negative values.

### Quantum Correction
A Quantum Correction Factor (QCF) reweights the classical spectrum. **The correct factor depends on which quantity you are computing**, and it is applied exactly once, to the raw spectrum.

#### IR (`IR.py`) — the harmonic QCF
For an IR spectrum from the classical dipole-current ACF:

$$ Q_{\text{harm}}(\omega) = \frac{\beta\hbar\omega}{1-e^{-\beta\hbar\omega}} = \frac{x}{1-e^{-x}}, \qquad x=\frac{\hbar\omega}{k_BT} $$

It satisfies detailed balance, is exact when the observable is linear in harmonic coordinates, and grows only **linearly** in $x$. Ramírez, López-Ciudad, Kumar & Marx (*J. Chem. Phys.* **121**, 3973 (2004)) compare QCFs for classical correlation functions and find the harmonic form performs best across most of their test cases. Implemented as `x / -expm1(-x)` for numerical stability, with the exact limit $Q\to 1$ as $x\to 0$.

#### VDOS (`vdos.py`) — off by default
A density of states is a property of the vibrational modes, not of their thermal population, so the classical VACF spectrum **already is** the VDOS. `quantum_correction` therefore defaults to `False`.

Setting it `True` applies the quantum/classical kinetic-energy ratio

$$ Q_{\text{KE}}(\omega) = \frac{x}{2}\coth\!\left(\frac{x}{2}\right) $$

which turns the output into a **quantum kinetic-energy-weighted vibrational spectrum — not a density of states**. The output file, plot title and metadata are relabelled accordingly.

### Nyquist limit
The highest physically representable wavenumber is

$$ \tilde\nu_{\text{Nyq}} = \frac{1}{2 c\, \Delta t_{\text{eff}}}, \qquad \Delta t_{\text{eff}} = \Delta t \times \texttt{nmeasure} $$

which is 66,713 cm⁻¹ for $\Delta t_{\text{eff}} = 0.25$ fs and 8,339 cm⁻¹ for 2 fs. A warning is issued if `plot_max_wavenumber` exceeds it. Note that `nmeasure > 1` reduces this limit proportionally, and that the effective timestep is used consistently for the velocity derivative, the frequency axis **and** the lag-window construction.

### Statistical uncertainty
`IR.py` reports the equivalent degrees of freedom $\nu$ and the approximate relative error $\sqrt{2/\nu}$ on band intensities. A dipole-current spectrum has only three Cartesian components to average over, whereas a VDOS averages over $3N$ atomic degrees of freedom, so **IR intensities converge far more slowly than the VDOS**. For a 5 ps trajectory at ~29 cm⁻¹ resolution the IR band error is roughly 20 %.

This is set by the trajectory length and the resolution you request — **not** by the estimator, which changes it by only ~12 %. The remedies are longer production trajectories, several independent trajectories, or block averaging with reported confidence intervals.

### Output metadata
Every output file carries a header recording: estimator, spectrum content (raw / quantum-weighted), velocity source, effective timestep, frames used, window kind and role, window/segment duration, overlap and segment count (or lags and `nfft`), bin spacing, effective resolution, Nyquist limit, quantum-correction choice, temperature, mass weighting, COM and PBC settings, masses, and — for VDOS — the sum-rule $T_{\text{eff}}$; for IR — charge mode, static charges, total charge and the estimated relative error.

The saved `.txt` is always the **raw, unfiltered** spectrum over the full frequency range. `plot_min_wavenumber` / `plot_max_wavenumber` affect the **plot only**.

### Workflow Structure
1.  **Input Reading**: XYZ (via ASE) and LAMMPS dump; NetCDF for VDOS only.
2.  **Preprocessing**: box-size inference, `nskip` / `nmeasure`, COM correction, PBC unwrapping.
3.  **Signal construction**: $\mathbf{v}_i(t)$, or $\mathbf{J}(t)=\sum_i q_i\mathbf{v}_i(t)$ for IR (with charge-neutrality validation).
4.  **Spectral estimation**: `acf` (biased vector ACF + decaying lag window + $2\mathrm{Re}[\cdot]-C(0)w(0)$) or `welch`.
5.  **Weighting**: explicit mass weighting (VDOS); quantum correction applied once, if enabled.
6.  **Validation**: velocity sum rule / $T_{\text{eff}}$ (VDOS), statistical uncertainty (IR).
7.  **Output**: raw data with full metadata (`.txt`) and a display-filtered plot (`.png`).

---

## Notes
1.  **Sampling Frequency**: To capture high-frequency modes such as the O–H stretch near 3600 cm⁻¹, the saved-frame interval must resolve the oscillation period (~9–10 fs). By Nyquist the sampling frequency must exceed twice the highest frequency of interest. A $\Delta t$ of 0.5 fs gives a Nyquist limit of 33,356 cm⁻¹, well beyond the vibrational range. Remember that `nmeasure` multiplies the effective interval.
2.  **Trajectory Sorting**: The module assumes the input trajectory is **SORTED** by atom ID (e.g. `dump_modify sort id` in LAMMPS) so atom indexing is consistent across frames.
3.  **Trajectory File Format**: The file must contain atom types. Cell boundaries are used if present, otherwise inferred. Note that PBC unwrapping assumes an orthorhombic cell.

For files containing cell boundaries, generate a trajectory in LAMMPS with:

    dump            3 all custom 1 ./traj-custom.xyz element x y z
    dump_modify     3 element C H
    dump_modify     3 sort id

Add `q` to the column list if you intend to use `charge_mode = dynamic`:

    dump            3 all custom 1 ./traj-custom.xyz element x y z q

Static mode does **not** require a `q` column. This produces:

    ITEM: TIMESTEP
    10454
    ITEM: NUMBER OF ATOMS
    59
    ITEM: BOX BOUNDS pp pp pp
    -5.7149802856090802e-02 1.9105149802856097e+01
    -4.1130782525593386e-02 9.9651307825256445e+00
    -2.7455019341267265e-02 9.2234550193412659e+00
    ITEM: ATOMS element x y z
    C 11.2212 3.17918 8.8797
    C 12.7296 2.80591 8.81307
    H 11.6504 4.87653 1.00421

If cell boundaries are not available they are inferred automatically, and the file can be generated with:

    dump            2 all xyz 1 ./traj.xyz
    dump_modify     2 element C H
    dump_modify     2 sort id

giving standard XYZ:

    59
    Atoms. Timestep: 10454
    C 11.2212 3.17918 8.8797
    C 12.7296 2.80591 8.81307
    C 10.8122 4.47872 0.365524
    H 11.6504 4.87653 1.00421

---

## Recommended validation sequence
1.  Run `vdos.py` and check the reported $T_{\text{eff}}$ against your thermostat temperature. Set `temperature` to the true value — it feeds the QCF.
2.  Run each script twice, once with `spectral_estimator = acf` and once with `welch`. Peak positions should agree to within the reported resolution, and integrated band intensities to within the reported statistical error.
3.  Confirm the printed total charge is zero within `charge_tolerance` before trusting any IR intensity.
4.  Check that `plot_max_wavenumber` is below the reported Nyquist limit and that no velocity-attenuation warning is printed.
