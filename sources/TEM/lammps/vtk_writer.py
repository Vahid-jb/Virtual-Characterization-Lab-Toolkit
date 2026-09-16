#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
VTK writer module for SAED data produces VTK files that can be directly read by ParaView and VisIt
"""
import numpy as np
import math
import os
from datetime import datetime

def write_saed_vtk(output_base, intensities, saed_params, output_index=0, echo=False):
    if echo:
        print(f"\n--- Writing SAED VTK file (index={output_index}) ---")
    
    lambda_val = saed_params['lambda_val']
    Kmax = saed_params['Kmax']
    Zone = saed_params['Zone'].copy()
    c = saed_params['c']
    dR_Ewald = saed_params['dR_Ewald']
    manual = saed_params['manual']
    prd_inv = saed_params['prd_inv']
    Knmax = saed_params['Knmax']
    Knmin = saed_params['Knmin']
    dK = saed_params['dK']
    store_tmp = saed_params['store_tmp']
    
    R_Ewald = 0.0
    zone_norm = math.sqrt(sum(z*z for z in Zone))
    full_space = (abs(Zone[0]) < 1e-10 and abs(Zone[1]) < 1e-10 and abs(Zone[2]) < 1e-10)
    
    if not full_space:
        R_Ewald = 1.0 / lambda_val
        if zone_norm > 1e-10:
            Rnorm = R_Ewald / zone_norm
            Zone = [z * Rnorm for z in Zone]
    
    Dim = [0, 0, 0]
    for i in range(3):
        if ((Knmin[i] > 0 and Knmax[i] > 0) or (Knmin[i] < 0 and Knmax[i] < 0)):
            Dim[i] = abs(Knmin[i]) + abs(Knmax[i])
        else:
            Dim[i] = abs(Knmin[i]) + abs(Knmax[i]) + 1
    
    if echo:
        print(f"VTK grid dimensions: {Dim}")
        print(f"Reciprocal space bounds: min={Knmin}, max={Knmax}")
        print(f"dK spacing: {dK}")
    
    output_filename = f"{output_base}_{output_index}.vtk"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            compute_id = saed_params.get('compute_id', 'SAED')
            f.write(f"# vtk DataFile Version 3.0 c_{compute_id}\n")
            f.write("Image data set\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {Dim[0]} {Dim[1]} {Dim[2]}\n")
            f.write("ASPECT_RATIO %g %g %g\n" % (dK[0], dK[1], dK[2]))
            f.write("ORIGIN %g %g %g\n" % (Knmin[0] * dK[0], Knmin[1] * dK[1],
                                           Knmin[2] * dK[2]))
            f.write(f"POINT_DATA {Dim[0] * Dim[1] * Dim[2]}\n")
            f.write("SCALARS intensity float\n")
            f.write("LOOKUP_TABLE default\n")
            
            NROW1 = 0
            total_points = 0
            
            for k in range(Knmin[2], Knmax[2] + 1):
                for j in range(Knmin[1], Knmax[1] + 1):
                    for i in range(Knmin[0], Knmax[0] + 1):
                        K = [
                            i * dK[0],
                            j * dK[1],
                            k * dK[2]
                        ]
                        dinv2 = K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
                        total_points += 1
                        
                        if dinv2 < Kmax * Kmax:
                            if full_space:
                                if NROW1 < len(intensities):
                                    f.write(f"{intensities[NROW1]:g}\n")
                                    NROW1 += 1
                                else:
                                    f.write("-1\n")
                            else:
                                r2 = sum((K[m] - Zone[m])**2 for m in range(3))
                                r = math.sqrt(r2)
                                if (r > (R_Ewald - dR_Ewald) and 
                                    r < (R_Ewald + dR_Ewald)):
                                    if NROW1 < len(intensities):
                                        f.write(f"{intensities[NROW1]:g}\n")
                                        NROW1 += 1
                                    else:
                                        f.write("-1\n")
                                else:
                                    f.write("-1\n")
                        else:
                            f.write("-1\n")
        
        if echo:
            print(f"Successfully wrote VTK file: {output_filename}")
            print(f"  Total points in grid: {total_points}")
            print(f"  Valid intensity points: {NROW1}/{len(intensities)}")
            print(f"  Ghost points (value = -1): {total_points - NROW1}")
        
        return output_filename
    
    except Exception as e:
        print(f"ERROR: Failed to write VTK file {output_filename}: {e}")
        raise

def write_saed_vtk_from_compute(compute_obj, output_base, output_index=0):
    if compute_obj.me == 0:  
        saed_params = {
            'lambda_val': compute_obj.lambda_val,
            'Kmax': compute_obj.Kmax,
            'Zone': compute_obj.Zone,
            'c': compute_obj.c,
            'dR_Ewald': compute_obj.dR_Ewald,
            'manual': compute_obj.manual,
            'prd_inv': compute_obj.prd_inv,
            'Knmax': compute_obj.Knmax,
            'Knmin': [0, 0, 0],  # Will be calculated
            'dK': compute_obj.dK,
            'store_tmp': compute_obj.store_tmp
        }
        
        full_space = (abs(compute_obj.Zone[0]) < 1e-10 and
                      abs(compute_obj.Zone[1]) < 1e-10 and
                      abs(compute_obj.Zone[2]) < 1e-10)

        if full_space:
            Knmax = [int(math.ceil(compute_obj.Kmax / compute_obj.dK[i]))
                     for i in range(3)]
            saed_params['Knmax'] = Knmax
            saed_params['Knmin'] = [-k for k in Knmax]
        elif compute_obj.nRows > 0:
            indices = compute_obj.store_tmp[:compute_obj.nRows]
            saed_params['Knmin'] = [
                int(np.min(indices[:, 0])),
                int(np.min(indices[:, 1])),
                int(np.min(indices[:, 2]))
            ]
            saed_params['Knmax'] = [
                int(np.max(indices[:, 0])),
                int(np.max(indices[:, 1])),
                int(np.max(indices[:, 2]))
            ]
        
        # Get intensities
        intensities = compute_obj.vector[:compute_obj.nRows]
        
        # Write VTK file
        return write_saed_vtk(
            output_base=output_base,
            intensities=intensities,
            saed_params=saed_params,
            output_index=output_index,
            echo=compute_obj.echo
        )
    
    return None

def write_multiple_vtk_files(compute_obj, output_base, intensities_list, 
                           ave_mode='one', nwindow=1, startstep=0):

    if compute_obj.me != 0:
        return []
    
    filenames = []
    nfiles = len(intensities_list)
    
    if ave_mode.lower() == 'one':
        for i, intensities in enumerate(intensities_list):
            filename = write_saed_vtk_from_compute(
                compute_obj, 
                output_base, 
                output_index=i
            )
            filenames.append(filename)
    
    elif ave_mode.lower() == 'running':
        if nfiles > 0:
            avg_intensities = np.zeros_like(intensities_list[0])
            for i, intensities in enumerate(intensities_list):
                avg_intensities = (avg_intensities * i + intensities) / (i + 1)
                
                filename = write_saed_vtk_from_compute(
                    compute_obj, 
                    output_base, 
                    output_index=i
                )
                filenames.append(filename)
    
    elif ave_mode.lower() == 'window':
        if nfiles > 0:
            window_size = min(nwindow, nfiles)
            
            for i in range(nfiles):
                start_idx = max(0, i - window_size + 1)
                window_data = intensities_list[start_idx:i+1]
                
                if window_data:
                    avg_intensities = np.mean(window_data, axis=0)
                    compute_obj.vector[:compute_obj.nRows] = avg_intensities
                    
                    filename = write_saed_vtk_from_compute(
                        compute_obj, 
                        output_base, 
                        output_index=i
                    )
                    filenames.append(filename)
    
    return filenames