# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_all

# Runtime libraries that must come from the host, not the bundle.
#
# PyInstaller puts _internal/ ahead of the system paths for the whole process,
# so a bundled copy of these shadows the host's everywhere -- including inside
# the graphics driver that Qt dlopens. The Linux bundle is built on Debian
# bullseye (GCC 10, up to GLIBCXX_3.4.28), while Mesa on a current distro needs
# GLIBCXX_3.4.30+; it then fails to load, GLX reports no usable FBConfig and Qt
# aborts with "Could not initialize GLX" before the window appears.
#
# Nothing we ship needs more than the build image provides, and every host at
# the bundle's glibc 2.31 floor (Debian 11, Ubuntu 20.04) already has
# GLIBCXX_3.4.28, so the host's copy is always new enough.
HOST_RUNTIME_LIBS = ('libstdc++.so', 'libgcc_s.so')


def strip_host_runtime_libs(analysis):
    """Drop HOST_RUNTIME_LIBS from an Analysis, Linux only.

    Only top-level entries: those are the ones on the loader's search path and
    hence the ones that shadow the host's. Copies a wheel vendors in its own
    subdirectory (numpy.libs/libgcc_s-<hash>.so.1) are loaded by that wheel
    alone and must stay.
    """
    if sys.platform != 'linux':
        return
    analysis.binaries = [
        entry for entry in analysis.binaries
        if os.path.dirname(entry[0])
        or not os.path.basename(entry[0]).startswith(HOST_RUNTIME_LIBS)
    ]


datas = [
    ('docs_site/site', 'docs_site/site'),
    # QSS icon assets (checkbox tick, combo and spin arrows). The destination
    # mirrors the package layout so the frozen VCL_utils/theme.py finds them
    # under _MEIPASS - see _theme_icon_dir() there. Deliberately unguarded: if
    # the directory ever goes missing the build must fail rather than ship a
    # bundle whose checkboxes have lost their tick.
    ('VCL_utils/assets/icons', 'VCL_utils/assets/icons'),
    # License texts ship inside the bundle (see THIRD_PARTY_NOTICES.md).
    ('LICENSE', '.'),
    ('THIRD_PARTY_NOTICES.md', '.'),
]
binaries = []
hiddenimports = [
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    # The tick and arrow assets are SVGs, which Qt decodes through the qsvg
    # image-format plugin; that plugin links libQt6Svg.
    "PySide6.QtSvg",
    "numpy",
    "scipy",
    "matplotlib",
    "matplotlib.backends.backend_qtagg",
]
tmp_ret = collect_all('ovito')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ase')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pymatgen')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# XRD-Kinematical reduces the saved CIF through pymatgen's SpacegroupAnalyzer,
# which needs spglib - a compiled extension collect_all('pymatgen') misses.
tmp_ret = collect_all('spglib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
try:
    tmp_ret = collect_all('periodictable')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass
try:
    tmp_ret = collect_all('scipy')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass
try:
    tmp_ret = collect_all('matplotlib')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass
# qtawesome ships the icon fonts the action buttons draw from, so its data
# files have to come along, not just the module.
try:
    tmp_ret = collect_all('qtawesome')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass




main = Analysis(
    ['converged_ui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
strip_host_runtime_libs(main)
main_pyz = PYZ(main.pure)

splash = Splash(
    'splash.png',
    binaries=main.binaries,
    datas=main.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True
)

main_exe = EXE(
    main_pyz,
    main.scripts,
    splash,
    splash.binaries,
    [],
    exclude_binaries=True,
    name='vcl_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

input_conv = Analysis(
    ['Input_Converter/input_convertor.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
input_conv_pyz = PYZ(input_conv.pure)

input_conv_exe = EXE(
    input_conv_pyz,
    input_conv.scripts,
    [],
    exclude_binaries=True,
    name='input_convertor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

ptm = Analysis(
    ['Structure_Analyzer/PTM.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
ptm_pyz = PYZ(ptm.pure)

ptm_exe = EXE(
    ptm_pyz,
    ptm.scripts,
    [],
    exclude_binaries=True,
    name='PTM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

lattice_detector = Analysis(
    ['Structure_Analyzer/lattice_detector.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
lattice_detector_pyz = PYZ(lattice_detector.pure)

lattice_detector_exe = EXE(
    lattice_detector_pyz,
    lattice_detector.scripts,
    [],
    exclude_binaries=True,
    name='lattice_detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

ir = Analysis(
    ['Vibrational_Analysis/IR.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
ir_pyz = PYZ(ir.pure)

ir_exe = EXE(
    ir_pyz,
    ir.scripts,
    [],
    exclude_binaries=True,
    name='IR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

vdos = Analysis(
    ['Vibrational_Analysis/vdos.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
vdos_pyz = PYZ(vdos.pure)

vdos_exe = EXE(
    vdos_pyz,
    vdos.scripts,
    [],
    exclude_binaries=True,
    name='vdos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

vtk_fig = Analysis(
    ['TEM/lammps/vtk_to_fig.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
vtk_fig_pyz = PYZ(vtk_fig.pure)

vtk_fig_exe = EXE(
    vtk_fig_pyz,
    vtk_fig.scripts,
    [],
    exclude_binaries=True,
    name='vtk_to_fig',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

xrd_deb = Analysis(
    ['XRD/XRD-Debye_Scattering.py'],
    pathex=['XRD'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
xrd_deb_pyz = PYZ(xrd_deb.pure)

xrd_deb_exe = EXE(
    xrd_deb_pyz,
    xrd_deb.scripts,
    [],
    exclude_binaries=True,
    name='XRD-Debye_Scattering',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

compute_saed = Analysis(
    ['TEM/lammps/compute_saed.py'],
    pathex=['TEM/lammps'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
compute_saed_pyz = PYZ(compute_saed.pure)

compute_saed_exe = EXE(
    compute_saed_pyz,
    compute_saed.scripts,
    [],
    exclude_binaries=True,
    name='compute_saed',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

xrd_rec = Analysis(
    ['XRD/XRD-ReciprocalSum.py'],
    pathex=['XRD'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
xrd_rec_pyz = PYZ(xrd_rec.pure)

xrd_rec_exe = EXE(
    xrd_rec_pyz,
    xrd_rec.scripts,
    [],
    exclude_binaries=True,
    name='XRD-ReciprocalSum',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

xrd_kin = Analysis(
    ['XRD/XRD-Kinematical.py'],
    pathex=['XRD'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
xrd_kin_pyz = PYZ(xrd_kin.pure)

xrd_kin_exe = EXE(
    xrd_kin_pyz,
    xrd_kin.scripts,
    [],
    exclude_binaries=True,
    name='XRD-Kinematical',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# `main` is already stripped above, before Splash reads its binaries.
for _analysis in (
    input_conv, ptm, lattice_detector, ir, vdos, vtk_fig,
    compute_saed, xrd_deb, xrd_kin, xrd_rec,
):
    strip_host_runtime_libs(_analysis)

coll = COLLECT(
    main_exe,
    main.binaries,
    main.datas,
    splash.binaries,
    input_conv_exe,
    input_conv.binaries,
    input_conv.datas,
    ptm_exe,
    ptm.binaries,
    ptm.datas,
    lattice_detector_exe,
    lattice_detector.binaries,
    lattice_detector.datas,
    ir_exe,
    ir.binaries,
    ir.datas,
    vdos_exe,
    vdos.binaries,
    vdos.datas,
    vtk_fig_exe,
    vtk_fig.binaries,
    vtk_fig.datas,
    compute_saed_exe,
    compute_saed.binaries,
    compute_saed.datas,
    xrd_deb_exe,
    xrd_deb.binaries,
    xrd_deb.datas,
    xrd_kin_exe,
    xrd_kin.binaries,
    xrd_kin.datas,
    xrd_rec_exe,
    xrd_rec.binaries,
    xrd_rec.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='converged_ui',
)

