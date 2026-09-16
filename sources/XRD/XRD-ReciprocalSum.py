#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Standalone XRD calculator based on LAMMPS compute_xrd code

"""

import os
import sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

try:
    from ase.io import read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not available. Limited file format support.")
    print("Install ASE with: pip install ase")

# Atomic scattering factor coefficients

ASFXRD = [
    [0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 0.049879, 2.20159, 0.001305],  # H
    [0.897661, 53.1368, 0.565616, 15.187, 0.415815, 186.576, 0.116973, 3.56709, 0.002389],  # He1-
    [0.8734, 9.1037, 0.6309, 3.3568, 0.3112, 22.9276, 0.178, 0.9821, 0.0064],  # He
    [1.1282, 3.9546, 0.7508, 1.0524, 0.6175, 85.3905, 0.4653, 68.261, 0.0377],  # Li
    [0.6968, 4.6237, 0.7888, 1.9557, 0.3414, 0.6316, 0.7029, 0.542, 0.0167],  # Li1+
    [1.5919, 43.6427, 1.1278, 1.8623, 0.5391, 103.483, 0.7029, 0.542, 0.0385],  # Be
    [6.2603, 0.0027, 0.8849, 0.8313, 0.7993, 2.2758, 0.1647, 5.1146, -6.1092],  # Be2+
    [2.0545, 23.2185, 1.3326, 1.021, 1.0979, 60.3498, 0.7068, 0.1403, -0.1932],  # B
    [2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 0.865, 51.6512, 0.2156],  # C
    [2.26069, 22.6907, 1.56165, 0.656665, 1.05075, 9.75618, 0.839259, 55.5949, 0.286977],  # Cval
    [12.2126, 0.0057, 3.1322, 9.8933, 2.0125, 28.9975, 1.1663, 0.5826, -11.529],  # N
    [3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.867, 32.9089, 0.2508],  # O
    [4.1916, 12.8573, 1.63969, 4.17236, 1.52673, -47.0179, 20.307, -0.01404, 21.9412],  # O1-
    [3.5392, 10.2825, 2.6412, 4.2944, 1.517, 0.2615, 1.0243, 26.1476, 0.2776],  # F
    [3.6322, 5.27756, 3.51057, 14.7353, 1.26064, 0.442258, 0.940706, 47.3437, 0.653396],  # F1-
    [3.9553, 8.4042, 3.1125, 3.4262, 1.4546, 0.2306, 1.1251, 21.7184, 0.3515],  # Ne
    [4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676],  # Na
    [3.2565, 2.6671, 3.9362, 6.1153, 1.3998, 0.2001, 1.0032, 14.039, 0.404],  # Na1+
    [5.4204, 2.8275, 2.1735, 79.2611, 1.2269, 0.3808, 2.3073, 7.1937, 0.8584],  # Mg
    [3.4988, 2.1676, 3.8378, 4.7542, 1.3284, 0.185, 0.8497, 10.1411, 0.4853],  # Mg2+
    [6.4202, 3.0387, 1.9002, 0.7426, 1.5936, 31.5472, 1.9646, 85.0886, 1.1151],  # Al
    [4.17448, 1.93816, 3.3876, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786],  # Al3+
    [6.2915, 2.4386, 3.0353, 32.3337, 1.9891, 0.6785, 1.541, 81.6937, 1.1407],  # Si
    [5.66269, 2.6652, 3.07164, 38.6634, 2.62446, 0.916946, 1.3932, 93.5458, 1.24707],  # Sival
    [4.43918, 1.64167, 3.20345, 3.43757, 1.19453, 0.2149, 0.41653, 6.65365, 0.746297],  # Si4+
    [6.4345, 1.9067, 4.1791, 27.157, 1.78, 0.526, 1.4908, 68.1645, 1.1149],  # P
    [6.9053, 1.4679, 5.2034, 22.2151, 1.4379, 0.2536, 1.5863, 56.172, 0.8669],  # S
    [11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574],  # Cl
    [18.2915, 0.0066, 7.2084, 1.1717, 6.5337, 19.5424, 2.3386, 60.4486, -16.378],  # Cl1-
    [7.4845, 0.9072, 6.7723, 14.8407, 0.6539, 43.8983, 1.6442, 33.3929, 1.4445],  # Ar
    [8.2186, 12.7949, 7.4398, 0.7748, 1.0519, 213.187, 0.8659, 41.6841, 1.4228],  # K
    [8.6266, 10.4421, 7.3873, 0.6599, 1.5899, 85.7484, 1.0211, 178.437, 1.3751],  # Ca
    [15.6348, -0.0074, 7.9518, 0.6089, 8.4372, 10.3116, 0.8537, 25.9905, -14.875],  # Ca2+
    [9.189, 9.0213, 7.3679, 0.5729, 1.6409, 136.108, 1.468, 51.3531, 1.3329],  # Sc
    [13.4008, 0.29854, 8.0273, 7.9629, 1.65943, -0.28604, 1.57936, 16.0662, -6.6667],  # Sc3+
    [9.7595, 7.8508, 7.3558, 0.5, 1.6991, 35.6338, 1.9021, 116.105, 1.2807],  # Ti
    [9.11423, 7.5243, 7.62174, 0.457585, 2.2793, 19.5361, 0.087899, 61.6558, 0.897155],  # Ti2+
    [17.7344, 0.22061, 8.73816, 7.04716, 5.25691, -0.15762, 1.92134, 15.9768, -14.652],  # Ti3+
    [19.5114, 0.178847, 8.23473, 6.67018, 2.01341, -0.29263, 1.5208, 12.9464, -13.28],  # Ti4+
    [10.2971, 6.8657, 7.3511, 0.4385, 2.0703, 26.8938, 2.0571, 102.478, 1.2199],  # V
    [10.106, 6.8818, 7.3541, 0.4409, 2.2884, 20.3004, 0.0223, 115.122, 1.2298],  # V2+
    [9.43141, 6.39535, 7.7419, 0.383349, 2.15343, 15.1908, 0.016865, 63.969, 0.656565],  # V3+
    [15.6887, 0.679003, 8.14208, 5.40135, 2.03081, 9.97278, -9.576, 0.940464, 1.7143],  # V5+
    [10.6406, 6.1038, 7.3537, 0.392, 3.324, 20.2626, 1.4922, 98.7399, 1.1832],  # Cr
    [9.54034, 5.66078, 7.7509, 0.344261, 3.58274, 13.3075, 0.509107, 32.4224, 0.616898],  # Cr2+
    [9.6809, 5.59463, 7.81136, 0.334393, 2.87603, 12.8288, 0.113575, 32.8761, 0.518275],  # Cr3+
    [11.2819, 5.3409, 7.3573, 0.3432, 3.0193, 17.8674, 2.2441, 83.7543, 1.0896],  # Mn
    [10.8061, 5.2796, 7.362, 0.3435, 3.5268, 14.343, 0.2184, 41.3235, 1.0874],  # Mn2+
    [9.84521, 4.91797, 7.87194, 0.294393, 3.56531, 10.8171, 0.323613, 24.1281, 0.393974],  # Mn3+
    [9.96253, 4.8485, 7.97057, 0.283303, 2.76067, 10.4852, 0.054447, 27.573, 0.251877],  # Mn4+
    [11.7695, 4.7611, 7.3573, 0.3072, 3.5222, 15.3535, 2.3045, 76.8805, 1.0369],  # Fe
    [11.0424, 4.6538, 7.374, 0.3053, 4.1346, 12.0546, 0.4399, 31.2809, 1.0097],  # Fe2+
    [11.1764, 4.6147, 7.3863, 0.3005, 3.3948, 11.6729, 0.0724, 38.5566, 0.9707],  # Fe3+
    [12.2841, 4.2791, 7.3409, 0.2784, 4.0034, 13.5359, 2.3488, 71.1692, 1.0118],  # Co
    [11.2296, 4.1231, 7.3883, 0.2726, 4.7393, 10.2443, 0.7108, 25.6466, 0.9324],  # Co2+
    [10.338, 3.90969, 7.88173, 0.238668, 4.76795, 8.35583, 0.725591, 18.3491, 0.286667],  # Co
    [12.8376, 3.8785, 7.292, 0.2565, 4.4438, 12.1763, 2.38, 66.3421, 1.0341],  # Ni
    [11.4166, 3.6766, 7.4005, 0.2449, 5.3442, 8.873, 0.9773, 22.1626, 0.8614],  # Ni2+
    [10.7806, 3.5477, 7.75868, 0.22314, 5.22746, 7.64468, 0.847114, 16.9673, 0.386044],  # Ni3+
    [13.338, 3.5828, 7.1676, 0.247, 5.6158, 11.3966, 1.6735, 64.8126, 1.191],  # Cu
    [11.9475, 3.3669, 7.3573, 0.2274, 6.2455, 8.6625, 1.5578, 25.8487, 0.89],  # Cu1+
    [11.8168, 3.37484, 7.11181, 0.244078, 5.78135, 7.9876, 1.14523, 19.897, 1.14431],  # Cu2+
    [14.0743, 3.2655, 7.0318, 0.2333, 5.1652, 10.3163, 2.41, 58.7097, 1.3041],  # Zn
    [11.9719, 2.9946, 7.3862, 0.2031, 6.4668, 7.0826, 1.394, 18.0995, 0.7807],  # Zn2+
    [15.2354, 3.0669, 6.7006, 0.2412, 4.3591, 10.7805, 2.9623, 61.4135, 1.7189],  # Ga
    [12.692, 2.81262, 6.69883, 0.22789, 6.06692, 6.36441, 1.0066, 14.4122, 1.53545],  # Ga3+
    [16.0816, 2.8509, 6.3747, 0.2516, 3.7068, 11.4468, 3.683, 54.7625, 2.1313],  # Ge
    [12.9172, 2.53718, 6.70003, 0.205855, 6.06791, 5.47913, 0.859041, 11.603, 1.45572],  # Ge4+
    [16.6723, 2.6345, 6.0701, 0.2647, 3.4313, 12.9479, 4.2779, 47.7972, 2.531],  # As
    [17.0006, 2.4098, 5.8196, 0.2726, 3.9731, 15.2372, 4.3543, 43.8163, 2.8409],  # Se
    [17.1789, 2.1723, 5.2358, 16.5796, 5.6377, 0.2609, 3.9851, 41.4328, 2.9557],  # Br
    [17.1718, 2.2059, 6.3338, 19.3345, 5.5754, 0.2871, 3.7272, 58.1535, 3.1776],  # Br1-
    [17.3555, 1.9384, 6.7286, 16.5623, 5.5493, 0.2261, 3.5375, 39.3972, 2.825],  # Kr
    [17.1784, 1.7888, 9.6435, 17.3151, 5.1399, 0.2748, 1.5292, 164.934, 3.4873],  # Rb
    [17.5816, 1.7139, 7.6598, 14.7957, 5.8981, 0.1603, 2.7817, 31.2087, 2.0782],  # Rb1+
    [17.5663, 1.5564, 9.8184, 14.0988, 5.422, 0.1664, 2.6694, 132.376, 2.5064],  # Sr
    [18.0874, 1.4907, 8.1373, 12.6963, 2.5654, -24.5651, 34.193, -0.0138, 41.4025],  # Sr2+
    [17.776, 1.4029, 10.2946, 12.8006, 5.72629, 0.125599, 3.26588, 104.354, 1.91213],  # Y
    [17.9268, 1.35417, 9.1531, 11.2145, 1.76795, -22.6599, 33.108, -0.01319, 40.2602],  # Y3+
    [17.8765, 1.27618, 10.948, 11.916, 5.41732, 0.117622, 3.65721, 87.6627, 2.06929],  # Zr
    [18.1668, 1.2148, 10.0562, 10.1483, 1.01118, 21.6054, -2.6479, -0.10276, 9.41454],  # Zr4+
    [17.6142, 1.18865, 12.0144, 11.766, 4.04183, 0.204785, 3.53346, 69.7957, 3.75591],  # Nb
    [19.8812, 0.019175, 18.0653, 1.13305, 11.0177, 10.1621, 1.94715, 28.3389, -12.912],  # Nb3+
    [17.9163, 1.12446, 13.3417, 0.028781, 10.799, 9.28206, 0.337905, 25.7228, -6.3934],  # Nb5+
    [3.7025, 0.2772, 17.2356, 1.0958, 12.8876, 11.004, 3.7429, 61.6584, 4.3875],  # Mo
    [21.1664, 0.014734, 18.2017, 1.03031, 11.7423, 9.53659, 2.30951, 26.6307, -14.421],  # Mo3+
    [21.0149, 0.014345, 18.0992, 1.02238, 11.4632, 8.78809, 0.740625, 23.3452, -14.316],  # Mo5+
    [17.8871, 1.03649, 11.175, 8.48061, 6.57891, 0.058881, 0, 0, 0.344941],  # Mo6+
    [19.1301, 0.864132, 11.0948, 8.14487, 4.64901, 21.5707, 2.71263, 86.8472, 5.40428],  # Tc
    [19.2674, 0.80852, 12.9182, 8.43467, 4.86337, 24.7997, 1.56756, 94.2928, 5.37814],  # Ru
    [18.5638, 0.847329, 13.2885, 8.37164, 9.32602, 0.017662, 3.00964, 22.887, -3.1892],  # Ru3+
    [18.5003, 0.844582, 13.1787, 8.12534, 4.71304, 0.36495, 2.18535, 20.8504, 1.42357],  # Ru4+
    [19.2957, 0.751536, 14.3501, 8.21758, 4.73425, 25.8749, 1.28918, 98.6062, 5.328],  # Rh
    [18.8785, 0.764252, 14.1259, 7.84438, 3.32515, 21.2487, -6.1989, -0.01036, 11.8678],  # Rh3+
    [18.8545, 0.760825, 13.9806, 7.62436, 2.53464, 19.3317, -5.6526, -0.0102, 11.2835],  # Rh4+
    [19.3319, 0.698655, 15.5017, 7.98929, 5.29537, 25.2052, 0.605844, 76.8986, 5.26593],  # Pd
    [19.1701, 0.696219, 15.2096, 7.55573, 4.32234, 22.5057, 0, 0, 5.2916],  # Pd2+
    [19.2493, 0.683839, 14.79, 7.14833, 2.89289, 17.9144, -7.9492, 0.005127, 13.0174],  # Pd4+
    [19.2808, 0.6446, 16.6885, 7.4726, 4.8045, 24.6605, 1.0463, 99.8156, 5.179],  # Ag
    [19.1812, 0.646179, 15.9719, 7.19123, 5.27475, 21.7326, 0.357534, 66.1147, 5.21572],  # Ag1+
    [19.1643, 0.645643, 16.2456, 7.18544, 4.3709, 21.4072, 0, 0, 5.21404],  # Ag2+
    [19.2214, 0.5946, 17.6444, 6.9089, 4.461, 24.7008, 1.6029, 87.4825, 5.0694],  # Cd
    [19.1514, 0.597922, 17.2535, 6.80639, 4.47128, 20.2521, 0, 0, 5.11937],  # Cd2+
    [19.1624, 0.5476, 18.5596, 6.3776, 4.2948, 25.8499, 2.0396, 92.8029, 4.9391],  # In
    [19.1045, 0.551522, 18.1108, 6.3247, 3.78897, 17.3595, 0, 0, 4.99635],  # In3+
    [19.1889, 5.8303, 19.1005, 0.5031, 4.4585, 26.8909, 2.4663, 83.9571, 4.7821],  # Sn
    [19.1094, 0.5036, 19.0548, 5.8378, 4.5648, 23.3752, 0.487, 62.2061, 4.7861],  # Sn2+
    [18.9333, 5.764, 19.7131, 0.4655, 3.4182, 14.0049, 0.0193, -0.7583, 3.9182],  # Sn4+
    [19.6418, 5.3034, 19.0455, 0.4607, 5.0371, 27.9074, 2.6827, 75.2825, 4.5909],  # Sb
    [18.9755, 0.467196, 18.933, 5.22126, 5.10789, 19.5902, 0.288753, 55.5113, 4.69626],  # Sb3+
    [19.8685, 5.44853, 19.0302, 0.467973, 2.41253, 14.1259, 0, 0, 4.69263],  # Sb5+
    [19.9644, 4.81742, 19.0138, 0.420885, 6.14487, 28.5284, 2.5239, 70.8403, 4.352],  # Te
    [20.1472, 4.347, 18.9949, 0.3814, 7.5138, 27.766, 2.2735, 66.8776, 4.0712],  # I
    [20.2332, 4.3579, 18.997, 0.3815, 7.8069, 29.5259, 2.8868, 84.9304, 4.0714],  # I1-
    [20.2933, 3.9282, 19.0298, 0.344, 8.9767, 26.4659, 1.99, 64.2658, 3.7118],  # Xe
    [20.3892, 3.569, 19.1062, 0.3107, 10.662, 24.3879, 1.4953, 213.904, 3.3352],  # Cs
    [20.3524, 3.552, 19.1278, 0.3086, 10.2821, 23.7128, 0.9615, 59.4565, 3.2791],  # Cs1+
    [20.3361, 3.216, 19.297, 0.2756, 10.888, 20.2073, 2.6959, 167.202, 2.7731],  # Ba
    [20.1807, 3.21367, 19.1136, 0.28331, 10.9054, 20.0558, 0.77634, 51.746, 3.02902],  # Ba2+
    [20.578, 2.94817, 19.599, 0.244475, 11.3727, 18.7726, 3.28719, 133.124, 2.14678],  # La
    [20.2489, 2.9207, 19.3763, 0.250698, 11.6323, 17.8211, 0.336048, 54.9453, 2.4086],  # La3+
    [21.1671, 2.81219, 19.7695, 0.226836, 11.8513, 17.6083, 3.33049, 127.113, 1.86264],  # Ce
    [20.8036, 2.77691, 19.559, 0.23154, 11.9369, 16.5408, 0.612376, 43.1692, 2.09013],  # Ce3+
    [20.3235, 2.65941, 19.8186, 0.21885, 12.1233, 15.7992, 0.144583, 62.2355, 1.5918],  # Ce4+
    [22.044, 2.77393, 19.6697, 0.222087, 12.3856, 16.7669, 2.82428, 143.644, 2.0583],  # Pr
    [21.3727, 2.6452, 19.7491, 0.214299, 12.1329, 15.323, 0.97518, 36.4065, 1.77132],  # Pr3+
    [20.9413, 2.54467, 20.0539, 0.202481, 12.4668, 14.8137, 0.296689, 45.4643, 1.24285],  # Pr4+
    [22.6845, 2.66248, 19.6847, 0.210628, 12.774, 15.885, 2.85137, 137.903, 1.98486],  # Nd
    [21.961, 2.52722, 19.9339, 0.199237, 12.12, 14.1783, 1.51031, 30.8717, 1.47588],  # Nd3+
    [23.3405, 2.5627, 19.6095, 0.202088, 13.1235, 15.1009, 2.87516, 132.721, 2.02876],  # Pm
    [22.5527, 2.4174, 20.1108, 0.185769, 12.0671, 13.1275, 2.07492, 27.4491, 1.19499],  # Pm3+
    [24.0042, 2.47274, 19.4258, 0.196451, 13.4396, 14.3996, 2.89604, 128.007, 2.20963],  # Sm
    [23.1504, 2.31641, 20.2599, 0.174081, 11.9202, 12.1571, 2.71488, 24.8242, 0.954586],  # Sm3+
    [24.6274, 2.3879, 19.0886, 0.1942, 13.7603, 13.7546, 2.9227, 123.174, 2.5745],  # Eu
    [24.0063, 2.27783, 19.9504, 0.17353, 11.8034, 11.6096, 3.87243, 26.5156, 1.36389],  # Eu2+
    [23.7497, 2.22258, 20.3745, 0.16394, 11.8509, 11.311, 3.26503, 22.9966, 0.759344],  # Eu3+
    [25.0709, 2.25341, 19.0798, 0.181951, 13.8518, 12.9331, 3.54545, 101.398, 2.4196],  # Gd
    [24.3466, 2.13553, 20.4208, 0.155525, 11.8708, 10.5782, 3.7149, 21.7029, 0.645089],  # Gd3+
    [25.8976, 2.24256, 18.2185, 0.196143, 14.3167, 12.6648, 2.95354, 115.362, 3.58324],  # Tb
    [24.9559, 2.05601, 20.3271, 0.149525, 12.2471, 10.0499, 3.773, 21.2773, 0.691967],  # Tb3+
    [26.507, 2.1802, 17.6383, 0.202172, 14.5596, 12.1899, 2.96577, 111.874, 4.29728],  # Dy
    [25.5395, 1.9804, 20.2861, 0.143384, 11.9812, 9.34972, 4.50073, 19.581, 0.68969],  # Dy3+
    [26.9049, 2.07051, 17.294, 0.19794, 14.5583, 11.4407, 3.63837, 92.6566, 4.56796],  # Ho
    [26.1296, 1.91072, 20.0994, 0.139358, 11.9788, 8.80018, 4.93676, 18.5908, 0.852795],  # Ho3+
    [27.6563, 2.07356, 16.4285, 0.223545, 14.9779, 11.3604, 2.98233, 105.703, 5.92046],  # Er
    [26.722, 1.84659, 19.7748, 0.13729, 12.1506, 8.36225, 5.17379, 17.8974, 1.17613],  # Er3+
    [28.1819, 2.02859, 15.8851, 0.238849, 15.1542, 10.9975, 2.98706, 102.961, 1.63929],  # Tm
    [27.3083, 1.78711, 19.332, 0.136974, 12.3339, 7.96778, 5.38348, 17.2922, 1.63929],  # Tm3+
    [28.6641, 1.9889, 15.4345, 0.257119, 15.3087, 10.6647, 2.98963, 100.417, 7.56672],  # Yb
    [28.1209, 1.78503, 17.6817, 0.15997, 13.3335, 8.18304, 5.14657, 20.39, 3.70983],  # Yb2+
    [27.8917, 1.73272, 18.7614, 0.13879, 12.6072, 7.64412, 5.47647, 16.8153, 2.26001],  # Yb3+
    [28.9476, 1.90182, 15.2208, 9.98519, 15.1, 0.261033, 3.71601, 84.3298, 7.97628],  # Lu
    [28.4628, 1.68216, 18.121, 0.142292, 12.8429, 7.33727, 5.59415, 16.3535, 2.97573],  # Lu3+
    [29.144, 1.83262, 15.1726, 9.5999, 14.7586, 0.275116, 4.30013, 72.029, 8.58154],  # Hf
    [28.8131, 1.59136, 18.4601, 0.128903, 12.7285, 6.76232, 5.59927, 14.0366, 2.39699],  # Hf4+
    [29.2024, 1.77333, 15.2293, 9.37046, 14.5135, 0.295977, 4.76492, 63.3644, 9.24354],  # Ta
    [29.1587, 1.50711, 18.8407, 0.116741, 12.8268, 6.31524, 5.38695, 12.4244, 1.78555],  # Ta5+
    [29.0818, 1.72029, 15.43, 9.2259, 14.4327, 0.321703, 5.11982, 57.056, 9.8875],  # W
    [29.4936, 1.42755, 19.3763, 0.104621, 13.0544, 5.93667, 5.06412, 11.1972, 1.01074],  # W6+
    [28.7621, 1.67191, 15.7189, 9.09227, 14.5564, 0.3505, 5.44174, 52.0861, 10.472],  # Re
    [28.1894, 1.62903, 16.155, 8.97948, 14.9305, 0.382661, 5.67589, 48.1647, 11.0005],  # Os
    [30.419, 1.37113, 15.2637, 6.84706, 14.7458, 0.165191, 5.06795, 18.003, 6.49804],  # Os4+
    [27.3049, 1.59279, 16.7296, 8.86553, 15.6115, 0.417916, 5.83377, 45.0011, 11.4722],  # Ir
    [30.4156, 1.34323, 15.862, 7.10909, 13.6145, 0.204633, 5.82008, 20.3254, 8.27903],  # Ir3+
    [30.7058, 1.30923, 15.5512, 6.71983, 14.2326, 0.167252, 5.53672, 17.4911, 6.96824],  # Ir4+
    [27.0059, 1.51293, 17.7639, 8.81174, 15.7131, 0.424593, 5.7837, 38.6103, 11.6883],  # Pt
    [29.8429, 1.32927, 16.7224, 7.38979, 13.2153, 0.263297, 6.35234, 22.9426, 9.85329],  # Pt2+
    [30.9612, 1.24813, 15.9829, 6.60834, 13.7348, 0.16864, 5.92034, 16.9392, 7.39534],  # Pt4+
    [16.8819, 0.4611, 18.5913, 8.6216, 25.5582, 1.4826, 5.86, 36.3956, 12.0658],  # Au
    [28.0109, 1.35321, 17.8204, 7.7395, 14.3359, 0.356752, 6.58077, 26.4043, 11.2299],  # Au1+
    [30.6886, 1.2199, 16.9029, 6.82872, 12.7801, 0.212867, 6.52354, 18.659, 9.0968],  # Au3+
    [20.6809, 0.545, 19.0417, 8.4484, 21.6575, 1.5729, 5.9676, 38.3246, 12.6089],  # Hg
    [25.0853, 1.39507, 18.4973, 7.65105, 16.8883, 0.443378, 6.48216, 28.2262, 12.0205],  # Hg1+
    [29.5641, 1.21152, 18.06, 7.05639, 12.8374, 0.284738, 6.89912, 20.7482, 10.6268],  # Hg2+
    [27.5446, 0.65515, 19.1584, 8.70751, 15.538, 1.96347, 5.52593, 45.8149, 13.1746],  # Tl
    [21.3985, 1.4711, 20.4723, 0.517394, 18.7478, 7.43463, 6.82847, 28.8482, 12.5258],  # Tl1+
    [30.8695, 1.1008, 18.3481, 6.53852, 11.9328, 0.219074, 7.00574, 17.2114, 9.8027],  # Tl3+
    [31.0617, 0.6902, 13.0637, 2.3576, 18.442, 8.618, 5.9696, 47.2579, 13.4118],  # Pb
    [21.7886, 1.3366, 19.5682, 0.488383, 19.1406, 6.7727, 7.01107, 23.8132, 12.4734],  # Pb2+
    [32.1244, 1.00566, 18.8003, 6.10926, 12.0175, 0.147041, 6.96886, 14.714, 8.08428],  # Pb4+
    [33.3689, 0.704, 12.951, 2.9238, 16.5877, 8.7937, 6.4692, 48.0093, 13.5782],  # Bi
    [21.8053, 1.2356, 19.5026, 6.24149, 19.1053, 0.469999, 7.10295, 20.3185, 12.4711],  # Bi3+
    [33.5364, 0.91654, 25.0946, 0.39042, 19.2497, 5.71414, 6.91555, 12.8285, 6.7994],  # Bi5+
    [34.6726, 0.700999, 15.4733, 3.55078, 13.1138, 9.55642, 7.02588, 47.0045, 13.677],  # Po
    [35.3163, 0.68587, 19.0211, 3.97458, 9.49887, 11.3824, 7.42518, 45.4715, 13.7108],  # At
    [35.5631, 0.6631, 21.2816, 4.0691, 8.0037, 14.0422, 7.4433, 44.2473, 13.6905],  # Rn
    [35.9299, 0.646453, 23.0547, 4.17619, 12.1439, 23.1052, 2.11253, 150.645, 13.7247],  # Fr
    [35.763, 0.616341, 22.9064, 3.87135, 12.4739, 19.9887, 3.21097, 142.325, 13.6211],  # Ra
    [35.215, 0.604909, 21.67, 3.5767, 7.91342, 12.601, 7.65078, 29.8436, 13.5431],  # Ra2+
    [35.6597, 0.589092, 23.1032, 3.65155, 12.5977, 18.599, 4.08655, 117.02, 13.5266],  # Ac
    [35.1736, 0.579689, 22.1112, 3.41437, 8.19216, 12.9187, 7.05545, 25.9443, 13.4637],  # Ac3+
    [35.5645, 0.563359, 23.4219, 3.46204, 12.7473, 17.8309, 4.80703, 99.1722, 13.4314],  # Th
    [35.1007, 0.555054, 22.4418, 3.24498, 9.78554, 13.4661, 5.29444, 23.9533, 13.376],  # Th4+
    [35.8847, 0.547751, 23.2948, 3.41519, 14.1891, 16.9235, 4.17287, 105.251, 13.4287],  # Pa
    [36.0228, 0.5293, 23.4128, 3.3253, 14.9491, 16.0927, 4.188, 100.613, 13.3966],  # U
    [35.5747, 0.52048, 22.5259, 3.12293, 12.2165, 12.7148, 5.37073, 26.3394, 13.3092],  # U3+
    [35.3715, 0.516598, 22.5326, 3.05053, 12.0291, 12.5723, 4.7984, 23.4582, 13.2671],  # U4+
    [34.8509, 0.507079, 22.7584, 2.8903, 14.0099, 13.1767, 1.21457, 25.2017, 13.1665],  # U6+
    [36.1874, 0.511929, 23.5964, 3.25396, 15.6402, 15.3622, 4.1855, 97.4908, 13.3573],  # Np
    [35.7074, 0.502322, 22.613, 3.03807, 12.9898, 12.1449, 5.43227, 25.4928, 13.2544],  # Np3+
    [35.5103, 0.498626, 22.5787, 2.96627, 12.7766, 11.9484, 4.92159, 22.7502, 13.2116],  # Np4+
    [35.0136, 0.48981, 22.7286, 2.81099, 14.3884, 12.33, 1.75669, 22.6581, 13.113],  # Np6+
    [36.5254, 0.499384, 23.8083, 3.26371, 16.7707, 14.9455, 3.47947, 105.98, 13.3812],  # Pu
    [35.84, 0.484938, 22.7169, 2.96118, 13.5807, 11.5331, 5.66016, 24.3992, 13.1991],  # Pu3+
    [35.6493, 0.481422, 22.646, 2.8902, 13.3595, 11.316, 5.18831, 21.8301, 13.1555],  # Pu4+
    [35.1736, 0.473204, 22.7181, 2.73848, 14.7635, 11.553, 2.28678, 20.9303, 13.0582],  # Pu6+
    [36.6706, 0.483629, 24.0992, 3.20647, 17.3415, 14.3136, 3.49331, 102.273, 13.3592],  # Am
    [36.6488, 0.465154, 24.4096, 3.08997, 17.399, 13.4346, 4.21665, 88.4834, 13.2887],  # Cm
    [36.7881, 0.451018, 24.7736, 3.04619, 17.8919, 12.8946, 4.23284, 86.003, 13.2754],  # Bk
    [36.9185, 0.437533, 25.1995, 3.00775, 18.3317, 12.4044, 4.24391, 83.7881, 13.2674],  # Cf
]

# Atom-type labels

XRDtypeList = [
    'H', 'He1-', 'He', 'Li', 'Li1+', 'Be',
    'Be2+', 'B', 'C', 'Cval', 'N', 'O',
    'O1-', 'F', 'F1-', 'Ne', 'Na', 'Na1+',
    'Mg', 'Mg2+', 'Al', 'Al3+', 'Si', 'Sival',
    'Si4+', 'P', 'S', 'Cl', 'Cl1-', 'Ar',
    'K', 'Ca', 'Ca2+', 'Sc', 'Sc3+', 'Ti',
    'Ti2+', 'Ti3+', 'Ti4+', 'V', 'V2+', 'V3+',
    'V5+', 'Cr', 'Cr2+', 'Cr3+', 'Mn', 'Mn2+',
    'Mn3+', 'Mn4+', 'Fe', 'Fe2+', 'Fe3+', 'Co',
    'Co2+', 'Co', 'Ni', 'Ni2+', 'Ni3+', 'Cu',
    'Cu1+', 'Cu2+', 'Zn', 'Zn2+', 'Ga', 'Ga3+',
    'Ge', 'Ge4+', 'As', 'Se', 'Br', 'Br1-',
    'Kr', 'Rb', 'Rb1+', 'Sr', 'Sr2+', 'Y',
    'Y3+', 'Zr', 'Zr4+', 'Nb', 'Nb3+', 'Nb5+',
    'Mo', 'Mo3+', 'Mo5+', 'Mo6+', 'Tc', 'Ru',
    'Ru3+', 'Ru4+', 'Rh', 'Rh3+', 'Rh4+', 'Pd',
    'Pd2+', 'Pd4+', 'Ag', 'Ag1+', 'Ag2+', 'Cd',
    'Cd2+', 'In', 'In3+', 'Sn', 'Sn2+', 'Sn4+',
    'Sb', 'Sb3+', 'Sb5+', 'Te', 'I', 'I1-',
    'Xe', 'Cs', 'Cs1+', 'Ba', 'Ba2+', 'La',
    'La3+', 'Ce', 'Ce3+', 'Ce4+', 'Pr', 'Pr3+',
    'Pr4+', 'Nd', 'Nd3+', 'Pm', 'Pm3+', 'Sm',
    'Sm3+', 'Eu', 'Eu2+', 'Eu3+', 'Gd', 'Gd3+',
    'Tb', 'Tb3+', 'Dy', 'Dy3+', 'Ho', 'Ho3+',
    'Er', 'Er3+', 'Tm', 'Tm3+', 'Yb', 'Yb2+',
    'Yb3+', 'Lu', 'Lu3+', 'Hf', 'Hf4+', 'Ta',
    'Ta5+', 'W', 'W6+', 'Re', 'Os', 'Os4+',
    'Ir', 'Ir3+', 'Ir4+', 'Pt', 'Pt2+', 'Pt4+',
    'Au', 'Au1+', 'Au3+', 'Hg', 'Hg1+', 'Hg2+',
    'Tl', 'Tl1+', 'Tl3+', 'Pb', 'Pb2+', 'Pb4+',
    'Bi', 'Bi3+', 'Bi5+', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ra2+', 'Ac', 'Ac3+', 'Th',
    'Th4+', 'Pa', 'U', 'U3+', 'U4+', 'U6+',
    'Np', 'Np3+', 'Np4+', 'Np6+', 'Pu', 'Pu3+',
    'Pu4+', 'Pu6+', 'Am', 'Cm', 'Bk', 'Cf',
]

XRDmaxType = 210

if len(XRDtypeList) != len(ASFXRD):
    raise RuntimeError(
        "Internal XRD table mismatch: %d type labels but %d ASF rows."
        % (len(XRDtypeList), len(ASFXRD)))
if len(ASFXRD) != XRDmaxType:
    raise RuntimeError(
        "Internal XRD table mismatch: expected %d rows (LAMMPS XRDmaxType) but found %d."
        % (XRDmaxType, len(ASFXRD)))
for _lab, _row in zip(XRDtypeList, ASFXRD):
    if len(_row) != 9:
        raise RuntimeError("Malformed 9-parameter ASF row for %r (got %d values)." % (_lab, len(_row)))
del _lab, _row


XRD_ALIASES = {
    "Co3+": 55,   
}

XRD_SUSPECT_ROWS = {
    146: "Tm: table coefficients give f(0) = 63.85 but Z(Tm) = 69 "
         "(upstream LAMMPS/ITC transcription defect, reproduced for equivalence)",
}


def check_asf_table(tolerance=0.35, verbose=False):
    try:
        from pymatgen.core.periodic_table import Element as _El

        def _z(sym):
            return _El(sym).Z
    except Exception:
        try:
            import periodictable as _pt

            def _z(sym):
                return _pt.elements.symbol(sym).number
        except Exception:
            return None 

    failures = []
    for idx, label in enumerate(XRDtypeList):
        if ('+' in label) or ('-' in label) or label.endswith('val'):
            continue
        try:
            Z = _z(label)
        except Exception:
            continue
        row = ASFXRD[idx]
        f0 = row[0] + row[2] + row[4] + row[6] + row[8]
        if abs(f0 - Z) > tolerance:
            failures.append((idx, label, f0, Z))
        elif verbose:
            print("  ok  %-6s idx %3d  f(0) = %8.3f  Z = %d" % (label, idx, f0, Z))
    return failures


class LAMMPSXRD:
    """XRD calculator"""

    def __init__(self, params):
        self.params = params
        self.ztype = []
        self.last_histogram = (None, None)
        self.dK = [0.0, 0.0, 0.0]
        self.Knmax = [0, 0, 0]
        self.valid_points = np.zeros((0, 3), dtype=np.int32)
        self.size_array_rows = 0
        self.size_array_cols = 2

        # Set defaults
        self.Min2Theta = 10.0  
        self.Max2Theta = 179.0  
        self.lambda_val = 1.5406  
        self.c = [1.0, 1.0, 1.0]  
        self.LP = 1  
        self.manual = False  
        self.echo = False  
        self.radflag = 1  

        self.prd_inv = [1.0, 1.0, 1.0]

        self.max_reciprocal_candidates = 20_000_000
        self.allow_large_reciprocal_grid = False

        self.compatibility_mode = 'strict_lammps'
        self.atom_type_mode = 'auto'
        self.atom_type_labels = []

        self.parse_params(params)

    def parse_params(self, params):
        """Parse input parameters with validation"""
        # Wavelength
        if 'wavelength' in params:
            try:
                self.lambda_val = float(params['wavelength'])
                if self.lambda_val <= 0:
                    print(f"Warning: Invalid wavelength {self.lambda_val}, using default 1.5406 A")
                    self.lambda_val = 1.5406
            except:
                print("Warning: Invalid wavelength parameter, using default 1.5406 A")

        # 2Theta range
        if '2Theta' in params:
            try:
                if isinstance(params['2Theta'], list) and len(params['2Theta']) >= 2:
                    self.Min2Theta = float(params['2Theta'][0])
                    self.Max2Theta = float(params['2Theta'][1])
                else:
                    parts = str(params['2Theta']).split(',')
                    if len(parts) >= 2:
                        self.Min2Theta = float(parts[0].strip())
                        self.Max2Theta = float(parts[1].strip())
                if self.Min2Theta <= 0 or self.Max2Theta >= 180 or self.Min2Theta >= self.Max2Theta:
                    print(f"Warning: Invalid 2Theta range ({self.Min2Theta}, {self.Max2Theta}), using default (10, 179)")
                    self.Min2Theta = 10.0
                    self.Max2Theta = 179.0
            except:
                print("Warning: Invalid 2Theta parameter, using default (10, 179)")

        # Resolution parameters
        if 'c' in params:
            try:
                if isinstance(params['c'], list) and len(params['c']) >= 3:
                    self.c = [float(x) for x in params['c'][:3]]
                else:
                    parts = str(params['c']).split(',')
                    if len(parts) >= 3:
                        self.c = [float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())]
                for i in range(3):
                    if self.c[i] <= 0:
                        print(f"Warning: Invalid resolution parameter c[{i}] = {self.c[i]}, using default 1.0")
                        self.c[i] = 1.0
            except:
                print("Warning: Invalid c parameter, using default [1.0, 1.0, 1.0]")


        # Periodic boundary conditions
        self.pbc_override = None  
        if 'pbc' in params:
            try:
                if isinstance(params['pbc'], list) and len(params['pbc']) >= 3:
                    self.pbc_override = [bool(int(x)) for x in params['pbc'][:3]]
                else:
                    parts = str(params['pbc']).split(',')
                    if len(parts) >= 3:
                        self.pbc_override = [bool(int(parts[0].strip())),
                                            bool(int(parts[1].strip())),
                                            bool(int(parts[2].strip()))]
                if self.echo and self.pbc_override is not None:
                    print(f"User-specified PBC: {self.pbc_override}")
            except Exception as e:
                print(f"Warning: Invalid PBC parameter format. Using structure file PBC. Error: {e}")

        # Manual box dimensions
        self.prd = None 
        if 'prd' in params:
            try:
                if isinstance(params['prd'], list) and len(params['prd']) >= 3:
                    self.prd = [float(x) for x in params['prd'][:3]]
                else:
                    parts = str(params['prd']).split(',')
                    if len(parts) >= 3:
                        self.prd = [float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())]
                if self.echo and self.prd is not None:
                    print(f"User-specified box dimensions: {self.prd} A")
            except Exception as e:
                print(f"Warning: Invalid prd parameter format. Using structure file dimensions. Error: {e}")
                self.prd = None


        # Lorentz-Polarization factor
        if 'LP' in params:
            try:
                self.LP = int(params['LP'])
                if self.LP not in [0, 1]:
                    print(f"Warning: Invalid LP value {self.LP}, must be 0 or 1, using default 1")
                    self.LP = 1
            except:
                print("Warning: Invalid LP parameter, using default 1")

        # Reciprocal-mesh cost guard
        if 'max_reciprocal_candidates' in params:
            try:
                self.max_reciprocal_candidates = int(float(params['max_reciprocal_candidates']))
            except Exception:
                print("Warning: invalid max_reciprocal_candidates, using default")
        if 'allow_large_reciprocal_grid' in params:
            self.allow_large_reciprocal_grid = bool(params['allow_large_reciprocal_grid'])

        # Compatibility mode
        cm = params.get('compatibility_mode', 'strict_lammps')
        cm = str(cm).strip().lower() if cm is not None else 'strict_lammps'
        if cm not in ('strict_lammps', 'relaxed'):
            print(f"Warning: unknown compatibility_mode '{cm}', using 'strict_lammps'")
            cm = 'strict_lammps'
        self.compatibility_mode = cm

        # Manual mode
        if 'manual' in params:
            self.manual = bool(params['manual'])

        # Echo mode
        if 'echo' in params:
            self.echo = bool(params['echo'])
        mode = params.get('atom_type_mode', 'auto')
        mode = str(mode).strip().lower() if mode is not None else 'auto'
        if mode not in ('auto', 'lammps_numeric', 'chemical_symbols'):
            print(f"Warning: unknown atom_type_mode '{mode}', using 'auto'")
            mode = 'auto'
        self.atom_type_mode = mode

        # Atom types - this is critical for the calculation
        if 'atom_types' not in params:
            print("Error: 'atom_types' parameter is required in the input file")
            print("Example: atom_types = Cu, Zn")
            print("It is an ORDERED list: entry i is the scattering-factor label for")
            print("LAMMPS atom type i+1, exactly as in the LAMMPS 'compute xrd' command.")
            print(f"Available atom types: {', '.join(XRDtypeList[:20])}...")
            sys.exit(1)

        atom_types = params['atom_types']
        if isinstance(atom_types, str):
            atom_types = [atom_types.strip()]
        elif isinstance(atom_types, (list, tuple)):
            atom_types = [str(x).strip() for x in atom_types]
        else:
            atom_types = [str(atom_types).strip()]

        self.atom_type_labels = atom_types
        self.ztype = [self.resolve_type_label(t) for t in atom_types]

    @staticmethod
    def resolve_type_label(label, strict=True):

        text = str(label).strip()
        for i, name in enumerate(XRDtypeList):
            if text.lower() == name.lower():
                if i in XRD_SUSPECT_ROWS:
                    print(f"Warning: {XRD_SUSPECT_ROWS[i]}")
                return i
        for alias, idx in XRD_ALIASES.items():
            if text.lower() == alias.lower():
                print(f"Note: '{text}' resolved through an alias to ASFXRD row {idx} "
                      f"(labelled '{XRDtypeList[idx]}' upstream in LAMMPS).")
                return idx
        if not strict:
            return None
        print(f"Error: Unknown atom type '{text}'")
        print("Labels must match the LAMMPS scattering-factor list exactly "
              "(for example 'Mo', 'Cu', 'Fe2+', 'Cval', 'Sival').")
        print("No fuzzy matching is performed: guessing an element from a structure-file "
              "tag silently substitutes the wrong form factor.")
        print(f"Available atom types: {', '.join(XRDtypeList)}")
        sys.exit(1)

    def map_atoms_to_asf_rows(self, atom_names):

        labels = [str(a).strip() for a in atom_names]
        numeric = all(lab.lstrip('+-').isdigit() for lab in labels) if labels else False

        mode = getattr(self, 'atom_type_mode', 'auto')
        if mode == 'auto':
            mode = 'lammps_numeric' if numeric else 'chemical_symbols'

        if mode == 'lammps_numeric':
            if not numeric:
                print("Error: atom_type_mode = lammps_numeric, but the structure file's "
                      "species column is not integer-valued.")
                sys.exit(1)
            n_types = len(self.ztype)
            rows = []
            for lab in labels:
                t = int(lab)
                if t < 1 or t > n_types:
                    print(f"Error: structure file contains LAMMPS atom type {t}, but "
                          f"'atom_types' supplies only {n_types} label(s): "
                          f"{self.atom_type_labels}")
                    print("Provide one label per numeric atom type, in type order.")
                    sys.exit(1)
                rows.append(self.ztype[t - 1])
            if self.echo:
                used = sorted(set(int(l) for l in labels))
                print("  Atom-type mapping (LAMMPS numeric): " + ", ".join(
                    f"type {t} -> {self.atom_type_labels[t - 1]}" for t in used))
            return rows

        # chemical_symbols: the file already carries scattering-factor labels.
        cache = {lab: self.resolve_type_label(lab) for lab in set(labels)}
        declared = {XRDtypeList[i] for i in self.ztype}
        present = {XRDtypeList[cache[l]] for l in set(labels)}
        extra = present - declared
        if extra:
            print(f"Warning: structure file contains species {sorted(extra)} that are not "
                  f"listed in 'atom_types' ({sorted(declared)}); they were resolved from "
                  f"the file's own labels.")
        if self.echo:
            print("  Atom-type mapping (chemical symbols): " + ", ".join(
                f"{l} -> {XRDtypeList[cache[l]]}" for l in sorted(set(labels))))
        return [cache[l] for l in labels]

    def infer_box_dimensions(self, positions):
        """
        Infer box dimensions from atomic positions when no cell info is available

        """
        if self.echo:
            print("Inferring box dimensions from atomic positions...")

        global_min = np.min(positions, axis=0)
        global_max = np.max(positions, axis=0)

        lengths = global_max - global_min
        min_box_size = 10.0  
        lengths = np.maximum(lengths, min_box_size)

        if self.echo:
            print(f"Inferred box dimensions: Lx={lengths[0]:.2f}A, Ly={lengths[1]:.2f}A, Lz={lengths[2]:.2f}A")
            print(f"Global position range: X[{global_min[0]:.2f}, {global_max[0]:.2f}], "
                  f"Y[{global_min[1]:.2f}, {global_max[1]:.2f}], Z[{global_min[2]:.2f}, {global_max[2]:.2f}]")

        return lengths

    def read_structure_file(self, filename):

        if ASE_AVAILABLE:
            if self.echo:
                print("Attempting to read structure file with ASE...")
            result = self.read_with_ase(filename)
            if result is not None:
                return result

        if filename.endswith('.data') or filename.endswith('.lmp'):
            if self.echo:
                print("ASE failed or not available. Trying LAMMPS data format...")
            result = self.read_lammps_data(filename)
            if result is not None:
                return result

        if self.echo:
            print("Trying XYZ format as fallback...")
        result = self.read_xyz_fallback(filename)
        if result is not None:
            return result

        raise ValueError(f"Could not read structure file '{filename}' with any available method.")


    def read_with_ase(self, filename):

        try:
            from ase.io import read
            atoms = read(filename)

            positions = atoms.get_positions()

            if hasattr(atoms, 'get_chemical_symbols'):
                atom_names = atoms.get_chemical_symbols()
            else:
                atom_names = []
                for atom in atoms:
                    if hasattr(atom, 'number'):
                        from ase.data import chemical_symbols
                        try:
                            atom_names.append(chemical_symbols[atom.number])
                        except:
                            atom_names.append("X")
                    else:
                        atom_names.append("X")

            if hasattr(atoms, 'get_cell'):
                cell = atoms.get_cell()
                off_diagonal_sum = (abs(cell[0,1]) + abs(cell[0,2]) +
                                   abs(cell[1,0]) + abs(cell[1,2]) +
                                   abs(cell[2,0]) + abs(cell[2,1]))
                if off_diagonal_sum > 1e-8:
                    print("Note: non-orthogonal (triclinic) cell detected in the structure file.")
                    print("      LAMMPS compute_xrd supports orthogonal boxes only; "
                          "init_calculation will reject it in strict mode.")
            else:
                cell = None

            if hasattr(atoms, 'get_pbc'):
                pbc = atoms.get_pbc()
            else:
                pbc = [False, False, False]

            num_atoms = len(atoms)

            if self.echo:
                print(f"ASE successfully read structure file:")
                print(f"  Atoms: {num_atoms}")
                print(f"  Cell dimensions: {cell[0,0]:.2f}, {cell[1,1]:.2f}, {cell[2,2]:.2f} A")
                print(f"  Periodic boundaries: {pbc}")
                print(f"  Atom types found: {set(atom_names)}")

            return np.array(positions), atom_names, cell, np.array(pbc), num_atoms

        except Exception as e:
            if self.echo:
                if str(e).find("Unknown') format") != -1:
                    print(f"ASE could not determine file format. Supported formats include:")
                    print(f"  XYZ, LAMMPS data, POSCAR/CONTCAR, CIF, VASP OUTCAR, etc.")
                else:
                    print(f"ASE error reading file: {e}")
            return None

    def read_lammps_data(self, filename):
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
                            num_atoms = int(parts[j-1])
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
                                atom_type = int(parts[0])
                                mass = float(parts[1])
                                if 90 < mass < 100:  
                                    atom_types[atom_type] = "Mo"
                                elif 50 < mass < 70:  
                                    atom_types[atom_type] = "Fe"
                                elif 20 < mass < 50: 
                                    atom_types[atom_type] = "Ca"
                                elif 10 < mass < 20:  
                                    atom_types[atom_type] = "C"
                                else:  
                                    atom_types[atom_type] = "H"
                        i += 1
                    continue

                # Parse atoms section
                elif "Atoms" in line:
                    i += 2  
                    atom_count = 0
                    while atom_count < num_atoms and i < len(lines):
                        atom_line = lines[i].strip()
                        if atom_line and not atom_line.startswith('#'):
                            parts = atom_line.split()
                            if len(parts) >= 5:
                                atom_id = int(parts[0])
                                atom_type = int(parts[1])
                                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                                positions.append([x, y, z])
                                element = atom_types.get(atom_type, "X")
                                atom_names.append(element)
                                atom_count += 1
                        i += 1
                    break

                i += 1

            if len(positions) == 0:
                return None

            if not all(have_box):
                missing = [t for t, h in zip('xyz', have_box) if not h]
                print(f"Warning: LAMMPS data file carries no {missing} box bounds; "
                      f"the cell is reported as unknown rather than guessed.")
                cell = None

            if self.echo:
                print(f"LAMMPS data file read successfully:")
                print(f"  Atoms: {len(positions)}")
                print(f"  Atom types found: {set(atom_names)}")
                if cell is not None:
                    print(f"  Box lengths: {cell[0, 0]:.4f}, {cell[1, 1]:.4f}, {cell[2, 2]:.4f} A")

            return np.array(positions), atom_names, cell, pbc, len(positions)

        except Exception as e:
            if self.echo:
                print(f"Error reading LAMMPS data file: {e}")
            return None

    def read_xyz_fallback(self, filename):
        if not os.path.exists(filename):
            return None

        try:
            with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            if len(lines) < 2:
                return None

            try:
                num_atoms = int(lines[0].strip())
                if num_atoms <= 0:
                    raise ValueError("Invalid number of atoms")
            except:
                num_atoms = 0
                for line in lines[2:]:
                    if line.strip() and len(line.split()) >= 4:
                        num_atoms += 1
                if num_atoms == 0:
                    return None

            positions = []
            atom_names = []

            for i in range(min(num_atoms, len(lines)-2)):
                parts = lines[i+2].split()
                if len(parts) < 4:
                    continue

                atom_name = parts[0].strip()
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except:
                    continue

                atom_names.append(atom_name)
                positions.append([x, y, z])

            if len(positions) == 0:
                return None

            cell, pbc = self._parse_extxyz_comment(lines[1] if len(lines) > 1 else "")

            if self.echo:
                print(f"XYZ file read successfully:")
                print(f"  Atoms: {len(positions)}")
                print(f"  Atom types found: {set(atom_names)}")
                if cell is None:
                    print("  Cell: not present in file (no extended-XYZ Lattice=\"...\")")
                else:
                    print(f"  Cell (from Lattice=): {np.linalg.norm(cell[0]):.4f}, "
                          f"{np.linalg.norm(cell[1]):.4f}, {np.linalg.norm(cell[2]):.4f} A")
                    print(f"  Periodic boundaries (from file): {pbc}")

            return np.array(positions), atom_names, cell, pbc, len(positions)

        except Exception as e:
            if self.echo:
                print(f"Error reading XYZ file: {e}")
            return None


    @staticmethod
    def _parse_extxyz_comment(comment):

        import re as _re
        m = _re.search(r'Lattice\s*=\s*"([^"]*)"', comment)
        if m is None:
            m = _re.search(r"Lattice\s*=\s*'([^']*)'", comment)
        if m is None:
            return None, None
        try:
            vals = [float(v) for v in m.group(1).replace(',', ' ').split()]
        except Exception:
            return None, None
        if len(vals) == 9:
            cell = np.array(vals, dtype=float).reshape(3, 3)
        elif len(vals) == 3:
            cell = np.diag(np.array(vals, dtype=float))
        else:
            return None, None

        pm = _re.search(r'pbc\s*=\s*"([^"]*)"', comment)
        if pm is None:
            pm = _re.search(r"pbc\s*=\s*'([^']*)'", comment)
        if pm is not None:
            toks = pm.group(1).replace(',', ' ').split()
            pbc = [str(t).strip().upper() in ('T', 'TRUE', '1', 'YES') for t in toks[:3]]
            while len(pbc) < 3:
                pbc.append(True)
        else:

            pbc = [True, True, True]
        return cell, pbc

    def init_calculation(self, positions, cell, pbc):
        """Initialize the XRD calculation with proper cell handling and early attribute initialization"""
        self.dK = [0.0, 0.0, 0.0]  
        self.prd_inv = [1.0, 1.0, 1.0]
        self.Knmax = [0, 0, 0]

        if self.pbc_override is not None:
            pbc = np.array(self.pbc_override)
            if self.echo:
                print(f"Using user-specified periodic boundaries: {pbc}")
        else:
            if self.echo:
                print(f"Using structure file periodic boundaries: {pbc}")

        self.Min2Theta = self.Min2Theta / 2.0 * math.pi / 180.0
        self.Max2Theta = self.Max2Theta / 2.0 * math.pi / 180.0
        self.radflag = 0  

        self.Kmax = 2 * math.sin(self.Max2Theta) / self.lambda_val

        box_dims = None
        have_cell = False

        if cell is not None:
            cell = np.asarray(cell, dtype=float)
            if cell.size == 9 and not np.allclose(cell, 0.0):
                cell = cell.reshape(3, 3)
                box_dims = np.array([np.linalg.norm(cell[i]) for i in range(3)])
                invalid_dims = [i for i, dim in enumerate(box_dims) if dim <= 1e-10]
                if invalid_dims:
                    if self.manual:
                        inferred = self.infer_box_dimensions(positions)
                        for i in invalid_dims:
                            box_dims[i] = inferred[i]
                        if self.echo:
                            print(f"Warning: cell lengths {invalid_dims} were zero; filled from "
                                  f"the coordinate span for reporting only (manual mode, "
                                  f"dK is set by c alone).")
                        have_cell = True
                    else:
                        raise ValueError(
                            f"Cell lengths in directions {invalid_dims} are zero or invalid. "
                            f"Auto mode needs the real periodic box because dK = c / L. "
                            f"Supply a structure file with valid cell metadata, or set "
                            f"manual = 1 to use a c-defined reciprocal mesh instead.")
                else:
                    have_cell = True

                off_diag = (abs(cell[0, 1]) + abs(cell[0, 2]) + abs(cell[1, 0]) +
                            abs(cell[1, 2]) + abs(cell[2, 0]) + abs(cell[2, 1]))
                scale = max(float(np.max(np.abs(cell))), 1e-30)
                if off_diag / scale > 1e-8:
                    msg = ("Non-orthogonal (triclinic) cell detected. LAMMPS compute_xrd "
                           "supports orthogonal boxes only, so reducing this cell to its "
                           "diagonal lengths would not reproduce any LAMMPS result.")
                    if self.compatibility_mode == 'strict_lammps':
                        raise ValueError(
                            msg + " Use an orthogonal box, or set compatibility_mode = relaxed "
                                  "(results are then NOT LAMMPS-equivalent), or manual = 1.")
                    print("WARNING: " + msg)
                    print("         compatibility_mode = relaxed: continuing with the "
                          "diagonal box lengths. Output is NOT LAMMPS-equivalent.")
                self.cell = cell

        if not have_cell:
            if self.manual:
                if positions is None:
                    raise ValueError("No cell information and no atomic positions available.")
                box_dims = self.infer_box_dimensions(positions)
                self.cell = np.diag(box_dims)
                if self.echo:
                    print("Note: no cell metadata in the structure file. Manual mode does not "
                          "use the box for the reciprocal mesh (dK = c), so the coordinate "
                          "span below is reported for information only.")
                    print(f"      Coordinate span: {box_dims}")
            else:
                raise ValueError(
                    "No valid cell information found in the structure file.\n"
                    "  Auto mode requires the real periodic box lengths, because the\n"
                    "  reciprocal mesh spacing is dK[i] = c[i] / L[i]. Options:\n"
                    "    * supply a structure file carrying cell metadata (LAMMPS data\n"
                    "      file, or extended XYZ with Lattice=\"...\"), or\n"
                    "    * install ASE for broader format support, or\n"
                    "    * set manual = 1 to use a c-defined reciprocal mesh that does\n"
                    "      not depend on the box at all.\n"
                    "  The box is NOT guessed from the coordinate span: that would change\n"
                    "  every peak position without warning.")

        if self.manual and self.prd is not None:
            if self.echo:
                print(f"Note: prd = {self.prd} recorded for reporting. In LAMMPS-compatible "
                      f"manual mode prd does NOT affect dK; c alone sets the mesh spacing.")
            box_dims = self.prd

        is_periodic = any(pbc)
        if not self.manual:
            if not is_periodic:
                raise ValueError("Error: Compute XRD must have at least one periodic boundary unless manual mode is enabled")

        if not self.manual:
            ave_inv = 0.0
            periodic_count = 0
            for i in range(3):
                if pbc[i] and box_dims[i] > 0:
                    self.prd_inv[i] = 1.0 / box_dims[i]
                    ave_inv += self.prd_inv[i]
                    periodic_count += 1
            if periodic_count > 0:
                ave_inv = ave_inv / periodic_count
                for i in range(3):
                    if not pbc[i] and box_dims[i] > 0:
                        self.prd_inv[i] = ave_inv
            else:
                self.prd_inv = [1.0 / box_dims[i] if box_dims[i] > 0 else 1.0 for i in range(3)]
        else:
            self.prd_inv = [1.0, 1.0, 1.0]
            if self.echo:
                print("Using manual reciprocal space mapping (all prd_inv = 1.0)")
            if any(pbc):
                print("Note: manual mode samples |F(K)|^2 on an arbitrary cubic mesh of "
                      "spacing c, which is generally NOT commensurate with a periodic "
                      "crystal's reciprocal lattice. Binned maxima will therefore sit "
                      "near, but not exactly at, the Bragg angles, and the raw weighted "
                      "histogram carries a |K|^2 node-density weighting. For a fully "
                      "periodic cell use manual = 0, where the mesh IS the reciprocal "
                      "lattice and peaks land on the Bragg angles exactly.")

        for i in range(3):
            self.dK[i] = self.prd_inv[i] * self.c[i]
            if self.dK[i] <= 0:
                raise ValueError(f"Invalid reciprocal spacing in direction {i}: {self.dK[i]}")

        self.Knmax = [0, 0, 0]
        for i in range(3):
            if self.dK[i] > 0:
                self.Knmax[i] = int(math.ceil(self.Kmax / self.dK[i]))
            else:
                self.Knmax[i] = 0

        nk = [int(self.Knmax[0]), int(self.Knmax[1]), int(self.Knmax[2])]
        n_candidates = (2 * nk[0] + 1) * (2 * nk[1] + 1) * (2 * nk[2] + 1)

        max_candidates = int(self.max_reciprocal_candidates)
        if n_candidates > max_candidates and not self.allow_large_reciprocal_grid:
            print("-----")
            print("Error: the requested reciprocal mesh is very large.")
            print(f"  Knmax                 : {nk}")
            print(f"  candidate mesh nodes  : {n_candidates:,}")
            print(f"  dK (1/Angstrom)       : {self.dK[0]:.6g}, {self.dK[1]:.6g}, {self.dK[2]:.6g}")
            print(f"  Kmax = 2 sin(theta_max)/lambda = {self.Kmax:.6g} 1/Angstrom")
            print("  Mesh node count scales as (Kmax/c)^3, so halving c multiplies the")
            print("  cost by eight.  Raise c, narrow the 2Theta range, or set")
            print("  allow_large_reciprocal_grid = 1 to proceed anyway.")
            print("-----")
            sys.exit(1)

        iv = np.arange(-nk[0], nk[0] + 1, dtype=np.int32)
        jv = np.arange(-nk[1], nk[1] + 1, dtype=np.int32)
        kv = np.arange(-nk[2], nk[2] + 1, dtype=np.int32)

        lam2 = self.lambda_val * self.lambda_val
        sin_min = math.sin(self.Min2Theta)
        sin_max = math.sin(self.Max2Theta)
        d2_lo = (2.0 * sin_min / self.lambda_val) ** 2
        d2_hi = (2.0 * sin_max / self.lambda_val) ** 2

        blocks = []
        n_accepted = 0
        kx2 = (iv * self.dK[0]) ** 2
        ky2 = (jv * self.dK[1]) ** 2
        kz2 = (kv * self.dK[2]) ** 2
        yz2 = (ky2[:, None] + kz2[None, :])
        for a, x2 in enumerate(kx2):
            dinv2 = x2 + yz2
            sel = (dinv2 * lam2 <= 4.0) & (dinv2 >= d2_lo) & (dinv2 <= d2_hi)
            if not sel.any():
                continue
            jj, kk = np.nonzero(sel)
            blk = np.empty((jj.size, 3), dtype=np.int32)
            blk[:, 0] = iv[a]
            blk[:, 1] = jv[jj]
            blk[:, 2] = kv[kk]
            blocks.append(blk)
            n_accepted += jj.size

        if blocks:
            self.valid_points = np.concatenate(blocks, axis=0)
        else:
            self.valid_points = np.zeros((0, 3), dtype=np.int32)
        nRows = int(self.valid_points.shape[0])

        if self.echo:
            print(f"  Candidate mesh nodes: {n_candidates:,} -> accepted: {nRows:,}")

        self.size_array_rows = nRows
        self.size_array_cols = 2


        if self.echo:
            print(f"-----")
            print(f"XRD Calculation Setup:")
            print(f"  Number of atoms: {len(positions)}")
            print(f"  Periodic boundaries: {pbc.tolist() if isinstance(pbc, np.ndarray) else pbc}")
            print(f"  Manual mode: {'enabled' if self.manual else 'disabled'}")
            print(f"  Box dimensions: {box_dims}")
            print(f"  Number of reciprocal lattice points: {nRows}")
            print(f"  Reciprocal spacing (k1,k2,k3): {self.dK[0]:.8f}, {self.dK[1]:.8f}, {self.dK[2]:.8f}")
            print(f"  prd_inv values: {self.prd_inv[0]:.8f}, {self.prd_inv[1]:.8f}, {self.prd_inv[2]:.8f}")
            print(f"  Wavelength: {self.lambda_val:.4f} A")
            print(f"  2theta range: {self.Min2Theta*2*180/math.pi:.1f} deg to {self.Max2Theta*2*180/math.pi:.1f} deg")
            print(f"  LP factor: {'enabled' if self.LP else 'disabled'}")
            print(f"  Atom types used: {[XRDtypeList[i] for i in self.ztype]}")
            print(f"-----")

        if nRows == 0:
            print("Warning: No reciprocal lattice points found within the specified 2theta range.")
            print("This may happen if:")
            print("  1. The 2theta range is too narrow")
            print("  2. The wavelength is too long/short")
            print("  3. The resolution parameters (c) are too coarse")
            print("  4. The cell is too large/small")
            print(f"  Current c values: {self.c}")
            print(f"  Manual mode: {'enabled' if self.manual else 'disabled'}")
            print(f"  Periodic boundaries: {pbc.tolist() if isinstance(pbc, np.ndarray) else pbc}")
            print(f"  Box dimensions: {box_dims}")

        return nRows

    def compute_xrd(self, positions, atom_names, cell, pbc):
        if self.echo:
            print("-----")
            print("Computing XRD pattern (vectorised over atoms)...")
            print(f"  Total reciprocal points: {self.size_array_rows}")
            print(f"  Total atoms: {len(positions)}")

        start_time = time.time()

        atom_rows = self.map_atoms_to_asf_rows(atom_names)

        natoms = len(positions)
        positions_arr = np.asarray(positions, dtype=np.float64)
        atom_rows_arr = np.asarray(atom_rows, dtype=np.int64)

        unique_rows, inverse = np.unique(atom_rows_arr, return_inverse=True)
        coeff = np.asarray([ASFXRD[r] for r in unique_rows], dtype=np.float64)  
        A_coef = coeff[:, 0:8:2]     
        B_coef = coeff[:, 1:8:2]     
        C_coef = coeff[:, 8]         

        Fvec = np.zeros((self.size_array_rows, 2))
        two_theta_deg = np.zeros(self.size_array_rows)

        pts = np.asarray(self.valid_points, dtype=np.float64)
        Kall = pts * np.asarray(self.dK, dtype=np.float64)[None, :]
        dinv2_all = np.einsum('ij,ij->i', Kall, Kall)
        sin_theta_all = 0.5 * np.sqrt(dinv2_all) * self.lambda_val
        np.clip(sin_theta_all, -1.0, 1.0, out=sin_theta_all)
        ang_all = np.arcsin(sin_theta_all)
        two_theta_deg[:] = np.degrees(2.0 * ang_all)

        progress_interval = max(1, self.size_array_rows // 10)

        for n in range(self.size_array_rows):
            K = Kall[n]
            SinTheta_lambda = 0.5 * math.sqrt(dinv2_all[n])
            ang = ang_all[n]
            CosTheta = math.cos(ang)
            Cos2Theta = math.cos(2.0 * ang)

            f_species = np.sum(A_coef * np.exp(-B_coef * SinTheta_lambda * SinTheta_lambda),
                               axis=1) + C_coef
            fj = f_species[inverse]

            inners = (2.0 * math.pi) * (positions_arr @ K)
            Fatom1 = float(np.dot(fj, np.cos(inners)))
            Fatom2 = float(np.dot(fj, np.sin(inners)))

            if self.LP == 1:
                SinTheta = SinTheta_lambda * self.lambda_val
                denominator = CosTheta * SinTheta * SinTheta
                if abs(denominator) < 1e-10:
                    sqrt_lp = 1.0
                else:
                    sqrt_lp = math.sqrt((1.0 + Cos2Theta * Cos2Theta) / denominator)
                Fatom1 *= sqrt_lp
                Fatom2 *= sqrt_lp

            Fvec[n, 0] = Fatom1
            Fvec[n, 1] = Fatom2

            if self.echo and (n + 1) % progress_interval == 0:
                progress = (n + 1) / self.size_array_rows * 100
                elapsed = time.time() - start_time
                est_total = elapsed / ((n + 1) / self.size_array_rows) if n > 0 else 0.0
                remaining = est_total - elapsed
                print(f"  Progress: {progress:5.1f}% ({n+1}/{self.size_array_rows} points) "
                      f"[ETA: {remaining:.1f}s]", end='\r')

        if self.echo:
            print("\nProgress: 100.0%")

        results = np.zeros((self.size_array_rows, self.size_array_cols))
        results[:, 0] = two_theta_deg
        results[:, 1] = (Fvec[:, 0] ** 2 + Fvec[:, 1] ** 2) / natoms

        elapsed_time = time.time() - start_time
        if self.echo:
            print(f"  Calculation completed in {elapsed_time:.2f} seconds")
            if elapsed_time > 0:
                print(f"  Throughput: ~{self.size_array_rows * natoms / (elapsed_time * 1e6):.1f}M "
                      f"atom-point evaluations/second")

        return results

    def get_atomic_scattering_factor(self, atom_type, SinTheta_lambda):
        if atom_type >= len(ASFXRD) or atom_type < 0:
            print(f"Warning: Invalid atom type index {atom_type}, using default H (index 0)")
            atom_type = 0

        coeffs = ASFXRD[atom_type]
        f_val = 0.0

        for i in range(0, 8, 2):
            A = coeffs[i]
            B = coeffs[i+1]
            f_val += A * math.exp(-B * SinTheta_lambda * SinTheta_lambda)

        f_val += coeffs[8]

        return f_val

    def save_results(self, results, filename, num_bins=250):

        valid_mask = (results[:, 0] > 0) & (results[:, 1] > 0)
        valid_results = results[valid_mask]

        if len(valid_results) == 0:
            print("Warning: No valid peaks found to save")
            return valid_results

        two_theta = valid_results[:, 0]
        intensities = valid_results[:, 1]

        min_angle = self.Min2Theta * 2 * 180 / math.pi
        max_angle = self.Max2Theta * 2 * 180 / math.pi

        hist, bin_edges = np.histogram(two_theta, bins=num_bins, range=(min_angle, max_angle), weights=intensities)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total_count = np.sum(hist)
        normalized = hist / total_count if total_count > 0 else np.zeros_like(hist)

        min_value = np.min(two_theta) if len(two_theta) > 0 else 0
        max_value = np.max(two_theta) if len(two_theta) > 0 else 0

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# LAMMPS compute_xrd-compatible reciprocal-space mesh summation\n")
            f.write(f"# Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Compatibility mode : {self.compatibility_mode}\n")
            f.write(f"# Cell restriction   : orthogonal only (as in LAMMPS compute_xrd)\n")
            f.write(f"# Reciprocal mode    : {'manual (dK = c)' if self.manual else 'auto (dK = c / L)'}\n")
            f.write(f"# Intensity          : |F(K)|^2 / N_atoms\n")
            f.write(f"# Histogram          : raw weighted SUM over reciprocal mesh nodes\n")
            f.write(f"#                      (not a normalised orientational powder average;\n")
            f.write(f"#                       node density per 2theta bin grows with |K|^2)\n")
            f.write(f"# Lorentz-polarization: {'enabled' if self.LP == 1 else 'disabled'}"
                    f"  LP = (1+cos^2 2t)/(cos t sin^2 t)\n")
            f.write(f"# Wavelength         : {self.lambda_val:.6f} Angstrom\n")
            f.write(f"# Reciprocal spacing : dK = {self.dK[0]:.8g}, {self.dK[1]:.8g}, {self.dK[2]:.8g} 1/Angstrom\n")
            f.write(f"# Mesh nodes used    : {self.size_array_rows}\n")
            f.write(f"# Atom-type mapping  : {self.atom_type_mode}; labels = {self.atom_type_labels}\n")
            f.write(f"# 2Theta range       : {min_angle:.4f} to {max_angle:.4f} degrees\n")
            f.write(f"# Number of bins     : {num_bins}\n")
            f.write("# Bin Coord Count Count/Total\n")

            for i in range(num_bins):
                bin_num = i + 1
                coord = bin_centers[i]
                count = hist[i]
                norm_count = normalized[i]
                f.write(f"{bin_num} {coord:.4f} {count:.6f} {norm_count:.6e}\n")

        if self.echo:
            print(f"  Results saved to {filename} with {num_bins} histogram bins")

        self.last_histogram = (hist, bin_centers)
        return valid_results

    def plot_results(self, results, filename, hist=None, bin_centers=None):
        """Plot the pattern.
        """
        valid_mask = (results[:, 0] > 0) & (results[:, 1] > 0)
        valid_results = results[valid_mask]

        if len(valid_results) == 0:
            print("Warning: No valid peaks to plot")
            return

        plt.figure(figsize=(12, 6), dpi=150)

        if hist is not None and bin_centers is not None and len(hist) > 1:
            plt.plot(bin_centers, hist, 'b-', linewidth=2.0,
                     label=f'Binned pattern ({len(hist)} bins)')
            smooth_sigma_deg = 0.0
            try:
                smooth_sigma_deg = float(self.params.get('plot_smoothing_sigma_deg', 0.0))
            except Exception:
                smooth_sigma_deg = 0.0
            if smooth_sigma_deg > 0.0 and len(bin_centers) > 2:
                dtheta = float(bin_centers[1] - bin_centers[0])
                if dtheta > 0:
                    sigma_pts = max(1e-6, smooth_sigma_deg / dtheta)
                    plt.plot(bin_centers, gaussian_filter1d(hist, sigma_pts),
                             'g-', linewidth=1.5, alpha=0.9,
                             label=f'Smoothed (sigma = {smooth_sigma_deg:g} deg, display only)')
            ymax = float(np.max(hist))
            if ymax > 0:
                scale = ymax / max(float(np.max(valid_results[:, 1])), 1e-300)
                plt.scatter(valid_results[:, 0], valid_results[:, 1] * scale,
                            s=6, c='r', alpha=0.35,
                            label=f'Mesh nodes ({len(valid_results):,}, rescaled)')
        else:
            order = valid_results[:, 0].argsort()
            valid_results = valid_results[order]
            plt.vlines(valid_results[:, 0], 0, valid_results[:, 1], colors='r',
                       linestyles='solid', linewidth=1.5, alpha=0.8,
                       label='Mesh node intensities')

        plt.xlabel('2theta (degrees)', fontsize=18, fontweight='bold')
        plt.ylabel('Intensity (a.u.)', fontsize=18, fontweight='bold')
        plt.title(f'XRD Pattern (lambda = {self.lambda_val:.4f} A)', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', width=1.5, length=8, pad=8)

        plt.tight_layout(pad=2.0)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        if self.echo:
            print(f"  Plot saved to {filename}")


def parse_input_file(filename):
    if not os.path.exists(filename):
        print(f"Error: Input file '{filename}' not found")
        sys.exit(1)

    params = {}
    line_num = 0

    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' not in line:
                print(f"Warning: Line {line_num} has no '=' separator, skipping")
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if '#' in value:
                value = value.split('#')[0].strip()

            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            if ',' in value and key not in ['wavelength', 'structure_file', 'output_file', 'plot_file']:
                items = [v.strip() for v in value.split(',')]
                converted_items = []
                for item in items:
                    try:
                        if '.' in item or 'e' in item.lower():
                            converted_items.append(float(item))
                        else:
                            converted_items.append(int(item))
                    except ValueError:
                        converted_items.append(item)
                value = converted_items

            elif value.lower() in ['true', 'yes', 'on', '1']:
                value = True
            elif value.lower() in ['false', 'no', 'off', '0']:
                value = False

            else:
                try:
                    if '.' in value or 'e' in value.lower():
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass

            params[key] = value

    return params

def self_test():

    print("=" * 70)
    print("SELF-TEST: XRD-ReciprocalSum (LAMMPS compute_xrd compatibility)")
    print("=" * 70)
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        else:
            fail += 1
            print(f"  FAIL  {name}  {detail}")

    check("label/row counts match LAMMPS XRDmaxType = 210",
          len(XRDtypeList) == len(ASFXRD) == XRDmaxType,
          f"{len(XRDtypeList)} labels, {len(ASFXRD)} rows")
    check("every row has 9 Cromer-Mann coefficients",
          all(len(r) == 9 for r in ASFXRD))

    failures = check_asf_table()
    if failures is None:
        print("  SKIP  f(0) = Z sum rule (no element database available)")
    else:
        expected = {(55, 'Co'), (146, 'Tm')}
        got = {(i, l) for i, l, _, _ in failures}
        check("only the two documented upstream LAMMPS defects break f(0) = Z",
              got == expected, f"failing rows: {sorted(got)}")

    def f_of(label, two_theta=0.0, lam=1.5406):
        idx = XRDtypeList.index(label)
        s = math.sin(math.radians(two_theta / 2.0)) / lam
        row = ASFXRD[idx]
        return sum(row[c] * math.exp(-row[c + 1] * s * s) for c in range(0, 8, 2)) + row[8]

    for el, Z in (("Mo", 42), ("Cu", 29), ("Zn", 30), ("Mn", 25), ("Fe", 26),
                  ("Sr", 38), ("Y", 39)):
        v = f_of(el)
        check(f"f(0) for {el} equals Z = {Z}", abs(v - Z) < 0.35, f"{v:.4f}")

    check("no Cr4+ / Cr5+ labels (the table has no such rows)",
          'Cr4+' not in XRDtypeList and 'Cr5+' not in XRDtypeList)
    check("Na b3 is the LAMMPS/ITC value 0.3136",
          ASFXRD[XRDtypeList.index('Na')][5] == 0.3136,
          f"{ASFXRD[XRDtypeList.index('Na')][5]}")

    calc = LAMMPSXRD({'atom_types': ['Mo', 'O'], 'echo': 0})
    rows = calc.map_atoms_to_asf_rows(['1', '2', '1'])
    check("LAMMPS numeric types map through ztype",
          rows == [XRDtypeList.index('Mo'), XRDtypeList.index('O'),
                   XRDtypeList.index('Mo')], f"{rows}")

    calc2 = LAMMPSXRD({'atom_types': 'Mo', 'echo': 0})
    calc2.Min2Theta = math.radians(5.0)
    calc2.Max2Theta = math.radians(45.0)
    calc2.size_array_rows = 3
    res = np.array([[20.0, 5.0], [30.0, 0.0], [40.0, 3.0]])
    try:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), 'xrd_selftest_absence.txt')
        out = calc2.save_results(res, tmp, num_bins=8)
        check("a zero-intensity reflection does not crash save_results",
              out.shape == (2, 2), f"returned {out.shape}")
    except Exception as e:
        check("a zero-intensity reflection does not crash save_results", False,
              f"{type(e).__name__}: {e}")

    for L in (10.0, 25.0):
        z = LAMMPSXRD({'atom_types': 'Mo', 'echo': 0, 'c': [1.0, 1.0, 1.0],
                       '2Theta': [10, 90], 'wavelength': 1.5406, 'manual': 0,
                       'pbc': [1, 1, 1]})
        z.init_calculation(np.zeros((2, 3)), np.eye(3) * L, [True, True, True])
        check(f"auto mode gives dK = c/L for L = {L}", abs(z.dK[0] - 1.0 / L) < 1e-15,
              f"dK = {z.dK[0]}")

    a_lat, ncell, lam = 3.147, 5, 1.5406
    pts = []
    for i in range(ncell):
        for j in range(ncell):
            for k in range(ncell):
                pts.append(((i) * a_lat, (j) * a_lat, (k) * a_lat))
                pts.append(((i + .5) * a_lat, (j + .5) * a_lat, (k + .5) * a_lat))
    pts = np.array(pts)
    box = ncell * a_lat
    xrd = LAMMPSXRD({'atom_types': 'Mo', 'echo': 0, 'c': [1.0, 1.0, 1.0],
                     '2Theta': [10, 90], 'wavelength': lam, 'manual': 0,
                     'pbc': [1, 1, 1], 'LP': 1})
    xrd.init_calculation(pts, np.eye(3) * box, [True, True, True])
    results = xrd.compute_xrd(pts, ['Mo'] * len(pts), np.eye(3) * box, [True] * 3)
    I = results[:, 1]
    tt = results[:, 0]
    imax = I.max()
    for hkl in ((1, 1, 0), (2, 0, 0), (2, 1, 1)):
        d = a_lat / math.sqrt(sum(v * v for v in hkl))
        ang = 2.0 * math.degrees(math.asin(lam / (2.0 * d)))
        near = np.abs(tt - ang) < 0.05
        check(f"bcc {hkl} present at 2theta = {ang:.3f} deg",
              near.any() and I[near].max() > 0.01 * imax,
              f"max I/Imax nearby = {(I[near].max() / imax if near.any() else 0):.3f}")
    for hkl in ((1, 0, 0), (1, 1, 1)):
        d = a_lat / math.sqrt(sum(v * v for v in hkl))
        ang = 2.0 * math.degrees(math.asin(lam / (2.0 * d)))
        near = np.abs(tt - ang) < 0.05
        val = I[near].max() / imax if near.any() else 0.0
        check(f"bcc {hkl} is a systematic absence at {ang:.3f} deg", val < 1e-12,
              f"I/Imax = {val:.3e}")

    print("=" * 70)
    print(f"SELF-TEST: {ok} passed, {fail} failed")
    print("=" * 70)
    return fail == 0


def main():
    """Main function"""
    if len(sys.argv) >= 2 and sys.argv[1] in ('--self-test', '-t'):
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) != 2:
        print("Usage: python lammps_xrd.py input.txt")
        print("\nExample input.txt file:")
        print("""
# Structure file (required)
structure_file = structure.xyz

# Output file (required)
output_file = xrd_pattern.txt

# X-ray wavelength in Angstroms (optional)
wavelength = 1.5406

# 2Theta range in degrees (optional)
2Theta = 10, 179

# Resolution parameters (optional)
c = 1, 1, 1

# Apply Lorentz-Polarization factor (optional)
LP = 1

# Manual mode (optional)
manual = 0

# Echo progress (optional)
echo = 1

# Generate plot (optional)
plot = 1

# Plot output file (optional)
plot_file = xrd_pattern.png

# Atom types (required - comma-separated list)
atom_types = Cu, Zn
""")
        sys.exit(1)

    input_file = sys.argv[1]
    params = parse_input_file(input_file)

    required_params = ['structure_file', 'output_file', 'atom_types']
    missing_params = [p for p in required_params if p not in params]

    if missing_params:
        print("Error: Missing required parameters in input file:")
        for param in missing_params:
            print(f"  - {param}")
        print("\nExample input file format:")
        print("""
structure_file = your_structure.xyz
output_file = xrd_results.txt
atom_types = Cu, Zn  # List all atom types in your structure
""")
        sys.exit(1)

    xrd_calc = LAMMPSXRD(params)

    try:
        positions, atom_names, cell, pbc, num_atoms = xrd_calc.read_structure_file(params['structure_file'])
        if xrd_calc.echo:
            print(f"\nStructure loaded successfully:")
            print(f"  Atoms: {num_atoms}")
            print(f"  Atom types found: {set(atom_names)}")
            print(f"  Atom types used: {[XRDtypeList[i] for i in xrd_calc.ztype]}")
            if cell is None:
                print("  Cell dimensions: not provided by the structure file")
            else:
                _c = np.asarray(cell, dtype=float).reshape(3, 3)
                print(f"  Cell dimensions: {np.linalg.norm(_c[0]):.2f}, "
                      f"{np.linalg.norm(_c[1]):.2f}, {np.linalg.norm(_c[2]):.2f} A")
            print(f"  Periodic boundaries: {pbc}")
    except Exception as e:
        print(f"Error reading structure file: {e}")
        sys.exit(1)

    try:
        nRows = xrd_calc.init_calculation(positions, cell, pbc)
        if nRows == 0:
            print("Error: No reciprocal lattice points found. Calculation cannot proceed.")
            print("Try adjusting the 2Theta range or checking your structure file.")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing calculation: {e}")
        sys.exit(1)

    try:
        results = xrd_calc.compute_xrd(positions, atom_names, cell, pbc)
    except Exception as e:
        print(f"Error during XRD calculation: {e}")
        sys.exit(1)

    try:
        num_bins = int(params.get('num_bins', params.get('numbins', 250)))
        if num_bins < 1:
            print(f"Warning: num_bins = {num_bins} is invalid, using 250")
            num_bins = 250
        final_results = xrd_calc.save_results(results, params['output_file'],
                                              num_bins=num_bins)
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    if params.get('plot', False):
        try:
            plot_file = params.get('plot_file', 'xrd_plot.png')
            if final_results.ndim == 2 and final_results.shape[1] >= 2:
                hist, bin_centers = getattr(xrd_calc, 'last_histogram', (None, None))
                xrd_calc.plot_results(final_results, plot_file,
                                      hist=hist, bin_centers=bin_centers)
            else:
                print("Warning: Results array has unexpected shape for plotting")
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            import traceback
            traceback.print_exc()

    print("\nXRD calculation completed successfully!")
    print(f"Results saved to: {params['output_file']}")
    if params.get('plot', False):
        print(f"Plot saved to: {params.get('plot_file', 'xrd_plot.png')}")

if __name__ == "__main__":
    main()