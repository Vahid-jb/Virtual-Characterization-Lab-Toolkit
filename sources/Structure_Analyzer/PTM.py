#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier, GrainSegmentationModifier
from ovito.pipeline import Pipeline, StaticSource
import sys
import numpy as np
import os
import argparse
import re
import json
from collections import Counter

# Machine-readable phase summary read by Structure_Analyzer/gui.py (which keeps its own copy of the name).
PHASE_MANIFEST = "ptm_phases.json"
MANIFEST_SCHEMA_VERSION = 1

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass
        try:
            self.log.write(message)
        except Exception:
            pass
    
    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass
    
    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='DXA Analysis Tool')
    parser.add_argument('input_txt', help='Path to input parameter file')
    args = parser.parse_args()

    params = {}
    known_keys = {'input_file', 'min_grain_size', 'RMSD_cutoff',
                  'input_file_type', 'trj_timestep'}
    canonical = {k.lower(): k for k in known_keys}

    try:
        with open(args.input_txt, 'r', encoding='utf-8') as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.split('#', 1)[0].strip()      # strip comments first
                if not line:
                    continue

                if line.startswith('--'):
                    line = line[2:]
                if '=' in line:
                    key, _, value = line.partition('=')
                else:
                    parts = line.split(None, 1)
                    key = parts[0]
                    value = parts[1] if len(parts) > 1 else ''

                key = key.strip()
                value = value.strip().strip('"').strip("'")

                canon = canonical.get(key.lower())
                if canon is None:
                    print(f"Warning: line {lineno}: unrecognised parameter "
                          f"'{key}' ignored.")
                    continue
                if canon in params:
                    print(f"Warning: line {lineno}: '{canon}' set more than "
                          f"once; using the last value ('{value}').")
                params[canon] = value

    except FileNotFoundError:
        print(f"Error: Parameter file '{args.input_txt}' not found.")
        sys.exit(1)

    if 'input_file_type' not in params:
        params['input_file_type'] = 'single'
    
    required_params = ['input_file', 'min_grain_size', 'RMSD_cutoff']
    for param in required_params:
        if param not in params:
            print(f"Error: Missing required parameter '{param}' in input file.")
            sys.exit(1)
    
    # Convert types
    try:
        params['min_grain_size'] = int(params['min_grain_size'])
        params['RMSD_cutoff'] = float(params['RMSD_cutoff'])
    except ValueError:
        print("Error: Invalid parameter types. min_grain_size must be integer, RMSD_cutoff must be float.")
        sys.exit(1)
    
    if params.get('input_file_type', 'single').lower() == 'traj':
        if 'trj_timestep' not in params or params['trj_timestep'] == "":
            print("Error: 'trj_timestep' must be provided when input_file_type is 'traj'.")
            sys.exit(1)
        try:
            params['trj_timestep'] = int(params['trj_timestep'])
        except ValueError:
            print("Error: Invalid trj_timestep. It must be an integer timestep value.")
            sys.exit(1)
    
    return params

def find_frame_index_by_timestep_xyz(filename, target_timestep, max_probe=None):

    try:
        with open(filename, 'r', encoding='utf-8') as fh:
            frame = 0
            while True:
                first = fh.readline()
                if not first:
                    break  
                comment = fh.readline()
                if not comment:
                    break
                m = re.search(r'Timestep[:\s]*([0-9]+)', comment, flags=re.IGNORECASE)
                if m:
                    try:
                        ts = int(m.group(1))
                        if ts == target_timestep:
                            return frame
                    except ValueError:
                        pass
                try:
                    natoms = int(first.strip())
                except Exception:
                    break
                for _ in range(natoms):
                    if not fh.readline():
                        break
                frame += 1
                if max_probe is not None and frame >= max_probe:
                    break
    except FileNotFoundError:
        return None
    return None

def get_element_counts(data, phase_mask):
    """Return {element: atom count} for the atoms in phase_mask, largest first, or None without element data."""
    if 'Particle Type' not in data.particles.keys() or data.particles.particle_types is None:
        return None

    particle_type_ids = data.particles['Particle Type'].array
    phase_type_ids = particle_type_ids[phase_mask]

    type_definitions = data.particles.particle_types.types
    type_name_map = {pt.id: pt.name for pt in type_definitions}

    element_counts = {}
    for type_id in phase_type_ids:
        element_name = type_name_map.get(type_id, f"Unknown_{type_id}")
        element_counts[element_name] = element_counts.get(element_name, 0) + 1

    return dict(sorted(element_counts.items(), key=lambda x: x[1], reverse=True))

def format_elemental_analysis(element_counts):
    if element_counts is None:
        return "N/A (no element data)"

    total_atoms_in_phase = sum(element_counts.values())
    if total_atoms_in_phase == 0:
        return "N/A (no atoms)"

    elemental_analysis_parts = []
    for element, count in element_counts.items():
        percentage = (count / total_atoms_in_phase) * 100
        elemental_analysis_parts.append(f"{percentage:.0f}% {element}")

    return ", ".join(elemental_analysis_parts)

def report_results(data, min_grain_size):
    
    print("\n--- Phase Distribution Report ---")
    if 'Structure Type' not in data.particles.keys():
        print("Error: 'Structure Type' property not found. PTM analysis may have failed.")
        return None, None

    struct_types = data.particles['Structure Type'].array
    unique_types, counts = np.unique(struct_types, return_counts=True)
    total_atoms = len(struct_types)

    type_map = {
        PolyhedralTemplateMatchingModifier.Type.OTHER.value: 'Other',
        PolyhedralTemplateMatchingModifier.Type.FCC.value: 'FCC',
        PolyhedralTemplateMatchingModifier.Type.HCP.value: 'HCP',
        PolyhedralTemplateMatchingModifier.Type.BCC.value: 'BCC',
        PolyhedralTemplateMatchingModifier.Type.ICO.value: 'ICO',
        PolyhedralTemplateMatchingModifier.Type.SC.value: 'SC',
        PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND.value: 'Cubic diamond',
        PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND.value: 'Hexagonal diamond',
        PolyhedralTemplateMatchingModifier.Type.GRAPHENE.value: 'Graphene',
    }

    detected_phases = {}
    orphan_atoms_count = 0
    
    phase_masks = {}
    
    for type_id in unique_types:
        phase_masks[type_id] = (struct_types == type_id)
    
    for i, (type_id, count) in enumerate(zip(unique_types, counts)):
        name = type_map.get(type_id, f'Unknown_{type_id}')
        percentage = (count / total_atoms) * 100
        
        phase_mask = phase_masks[type_id]
        element_counts = get_element_counts(data, phase_mask)
        elemental_analysis = format_elemental_analysis(element_counts)
        
        detected_phases[i+1] = {
            'id': int(type_id),
            'name': name,
            'count': int(count),
            'percentage': float(percentage),
            'composition': element_counts or {},
            'file': None,
        }
        
        phase_str = f"{i+1}: {name:<20}"
        atoms_str = f"Atoms: {count:<10}"
        perc_str = f"Percentage: {percentage:>6.2f}%"
        elem_str = f"elemental_analysis {elemental_analysis}"
        
        print(f"{phase_str} {atoms_str} {perc_str} {elem_str}")
        
        if type_id == PolyhedralTemplateMatchingModifier.Type.OTHER.value:
            orphan_atoms_count = count

    grain_table = data.tables.get('grains')
    num_grains = 0

    if grain_table is not None and 'Grain Identifier' in grain_table and 'Grain Size' in grain_table:
        grain_ids = grain_table['Grain Identifier'][...]
        grain_sizes = grain_table['Grain Size'][...]
        
        filtered_indices = grain_sizes >= min_grain_size
        filtered_grain_ids = grain_ids[filtered_indices]
        filtered_grain_sizes = grain_sizes[filtered_indices]
        
        num_grains = len(filtered_grain_ids)
        print(f"\nNumber of grains found (>= {min_grain_size} atoms): {num_grains}")
        
        if num_grains > 0:
            print("\nGrain Details:")
            if 'Mean Orientation' in grain_table:
                grain_orientations = grain_table['Mean Orientation'][...]
                filtered_grain_orientations = grain_orientations[filtered_indices]
                
                print("{:<10} {:<15} {:<50}".format('ID', 'Atoms', 'Mean Orientation (Quaternion)'))
                for grain_id, num_atoms_in_grain, orientation in zip(filtered_grain_ids, filtered_grain_sizes, filtered_grain_orientations):
                    orient_str = f"({orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f})"
                    print(f"{int(grain_id):<10} {int(num_atoms_in_grain):<15} {orient_str:<50}")
            else:
                print("{:<10} {:<15}".format('ID', 'Atoms'))
                for grain_id, num_atoms_in_grain in zip(filtered_grain_ids, filtered_grain_sizes):
                    print(f"{int(grain_id):<10} {int(num_atoms_in_grain):<15}")
        else:
            print("No grains were found that meet the specified minimum size.")
    else:
        print("No grains were found that meet the specified minimum size.")

    print(f"Total number of orphan atoms (non-crystalline): {orphan_atoms_count}")
    
    return detected_phases, orphan_atoms_count

def write_summary_file(filename, rmsd, min_grain_size, num_orphan, data):
    with open(filename, 'w') as f:
        f.write("--- Analysis Summary Report ---\n\n")
        f.write(f"RMSD Cutoff: {rmsd}\n")
        f.write(f"Minimum Grain Size: {min_grain_size} atoms\n")
        f.write(f"Total number of orphan atoms: {num_orphan}\n\n")

        f.write("--- Grain Details ---\n")
        grain_table = data.tables.get('grains')
        if grain_table is not None and 'Grain Identifier' in grain_table and 'Grain Size' in grain_table:
            grain_ids = grain_table['Grain Identifier'][...]
            grain_sizes = grain_table['Grain Size'][...]
            
            filtered_indices = grain_sizes >= min_grain_size
            filtered_grain_ids = grain_ids[filtered_indices]
            filtered_grain_sizes = grain_sizes[filtered_indices]
            
            if 'Mean Orientation' in grain_table:
                grain_orientations = grain_table['Mean Orientation'][...]
                filtered_grain_orientations = grain_orientations[filtered_indices]
                
                f.write("{:<10} {:<15} {:<50}\n".format('Grain ID', 'Atoms', 'Mean Orientation (Quaternion)'))
                for grain_id, num_atoms_in_grain, orientation in zip(filtered_grain_ids, filtered_grain_sizes, filtered_grain_orientations):
                    orient_str = f"({orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f})"
                    f.write(f"{int(grain_id):<10} {int(num_atoms_in_grain):<15} {orient_str:<50}\n")
            else:
                f.write("{:<10} {:<15}\n".format('Grain ID', 'Atoms'))
                for grain_id, num_atoms_in_grain in zip(filtered_grain_ids, filtered_grain_sizes):
                    f.write(f"{int(grain_id):<10} {int(num_atoms_in_grain):<15}\n")
        else:
            f.write("No grain data available for summary.\n")

def write_phase_manifest(filename, detected_phases, total_atoms, ptm_modifier, input_file, rmsd_cutoff):
    """Write the machine-readable phase summary the GUI uses for its 3D phase view."""
    # OVITO's display color for each PTM structure type, keyed by type ID
    structure_colors = {int(s.id): [float(c) for c in s.color] for s in ptm_modifier.structures}

    phases = []
    for key in sorted(detected_phases):
        phase = detected_phases[key]
        phases.append({
            "name": phase['name'],
            "structure_type": phase['id'],
            "count": phase['count'],
            "percentage": phase['percentage'],
            "composition": phase['composition'],
            "color": structure_colors.get(phase['id'], [0.7, 0.7, 0.7]),
            "file": phase['file'],
        })

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_file": input_file,
        "rmsd_cutoff": rmsd_cutoff,
        "total_atoms": int(total_atoms),
        "phases": phases,
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def main():
    params = parse_arguments()
    input_file = params['input_file']
    min_grain_size = params['min_grain_size']
    rmsd_cutoff = params['RMSD_cutoff']
    input_file_type = params.get('input_file_type', 'single').lower()
    trj_timestep = params.get('trj_timestep', None)
    
    log_filename = "structure_analysis.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"Structure analysis started with parameters:")
    print(f"  Input file: {input_file}")
    print(f"  Input file type: {input_file_type}")
    if input_file_type == 'traj':
        print(f"  Trajectory timestep requested: {trj_timestep}")
    print(f"  Min grain size: {min_grain_size}")
    print(f"  RMSD cutoff: {rmsd_cutoff}\n")
    
    try:
        pipeline = import_file(input_file)
    except FileNotFoundError:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Error importing file '{input_file}': {e}")
        sys.exit(1)

    frame_index = None
    if input_file_type == 'traj':
        num_frames = None
        if hasattr(pipeline.source, 'num_frames'):
            try:
                num_frames = int(pipeline.source.num_frames)
            except Exception:
                num_frames = None

        found = False
        if num_frames is None:
            max_probe = 2001
            for i in range(max_probe):
                try:
                    d = pipeline.compute(i)
                except Exception:
                    break
                ts = d.attributes.get('Timestep') if hasattr(d, 'attributes') else None
                if ts == trj_timestep:
                    frame_index = i
                    found = True
                    break
            if not found:
                frame_index = find_frame_index_by_timestep_xyz(input_file, trj_timestep, max_probe)
                if frame_index is None:
                    sys.stdout = logger.terminal
                    logger.close()
                    print(f"Error: Could not find timestep {trj_timestep} in trajectory (probed {max_probe} frames).")
                    sys.exit(1)
        else:
            for i in range(num_frames):
                try:
                    d = pipeline.compute(i)
                except Exception:
                    continue
                ts = d.attributes.get('Timestep') if hasattr(d, 'attributes') else None
                if ts == trj_timestep:
                    frame_index = i
                    found = True
                    break
            if not found:
                frame_index = find_frame_index_by_timestep_xyz(input_file, trj_timestep, max_probe=num_frames)
                if frame_index is None:
                    sys.stdout = logger.terminal
                    logger.close()
                    print(f"Error: Timestep {trj_timestep} not found in trajectory (checked {num_frames} frames).")
                    sys.exit(1)

    try:
        if frame_index is None:
            data = pipeline.compute()
        else:
            data = pipeline.compute(frame_index)
    except Exception as e:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Error computing initial data: {e}")
        sys.exit(1)

    num_atoms = data.particles.count
    
    has_particle_types = data.particles.particle_types is not None
    if has_particle_types:
        particle_types = data.particles.particle_types
        element_names = [pt.name for pt in particle_types.types]
        element_ids = [pt.id for pt in particle_types.types]
        print(f"Element types found: {', '.join([f'{name} (ID: {id})' for name, id in zip(element_names, element_ids)])}")
    
    print(f"\nSuccessfully loaded file: '{input_file}'")
    if frame_index is not None:
        print(f"Using trajectory frame index: {frame_index} (Timestep: {trj_timestep})")
    print(f"Number of atoms: {num_atoms}\n")

    pipeline.modifiers.clear()
    
    ptm_modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff)
    if hasattr(ptm_modifier, 'output_orientation'):
        ptm_modifier.output_orientation = True
    elif hasattr(ptm_modifier, 'calculate_orientations'):
        ptm_modifier.calculate_orientations = True
    pipeline.modifiers.append(ptm_modifier)

    grain_modifier = GrainSegmentationModifier()

    grain_modifier.algorithm = GrainSegmentationModifier.Algorithm.GraphClusteringAuto

    grain_modifier.orphan_adoption = False

    grain_modifier.min_grain_size = min_grain_size

    pipeline.modifiers.append(grain_modifier)
    
    try:
        if frame_index is None:
            last_computed_data = pipeline.compute()
        else:
            last_computed_data = pipeline.compute(frame_index)
    except Exception as e:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Analysis failed: {e}")
        sys.exit(1)
    
    detected_phases, num_orphan = report_results(last_computed_data, min_grain_size)
    
    if detected_phases is None:
        sys.stdout = logger.terminal
        logger.close()
        print("Analysis failed. Please check your input file and OVITO installation.")
        sys.exit(1)
    
    print("\n--- Exporting ALL detected phases ---")
    data_for_export = last_computed_data
        
    if 'Structure Type' in data_for_export.particles.keys():
        struct_types = data_for_export.particles['Structure Type'].array
        
        has_particle_type_prop = 'Particle Type' in data_for_export.particles.keys()
        
        for choice in detected_phases:
            selected_phase = detected_phases[choice]
            selection_mask = (struct_types == selected_phase['id'])
            
            if np.any(selection_mask):
                output_file = f"parent_phase_{selected_phase['name'].lower().replace(' ', '_')}.xyz"
                
                new_data = ovito.data.DataCollection()
                particles = ovito.data.Particles()
                
                positions = data_for_export.particles.position.array[selection_mask]
                particles.create_property('Position', data=positions)
                
                if has_particle_type_prop:
                    particle_type_ids = data_for_export.particles['Particle Type'].array[selection_mask]
                    particle_type_prop = particles.create_property('Particle Type', data=particle_type_ids)
                    
                    if data_for_export.particles.particle_types is not None:
                        particle_type_prop.types = data_for_export.particles.particle_types.types
                
                particles.create_property('Structure Type', 
                                         data=data_for_export.particles['Structure Type'].array[selection_mask])
                
                new_data.particles = particles
                new_data.cell = data_for_export.cell
                
                temp_pipeline = Pipeline(source=StaticSource(data=new_data))
                
                if has_particle_type_prop and data_for_export.particles.particle_types is not None:
                    export_file(
                        temp_pipeline.compute(),
                        output_file,
                        'xyz',
                        columns=['Particle Type', 'Position.X', 'Position.Y', 'Position.Z']
                    )
                else:
                    export_file(
                        temp_pipeline.compute(),
                        output_file,
                        'xyz',
                        columns=['Position.X', 'Position.Y', 'Position.Z']
                    )
                    print(f"Warning: Exported '{output_file}' without element labels.")
                
                print(f"Exported {np.sum(selection_mask)} atoms of phase '{selected_phase['name']}' to '{output_file}'.")
                selected_phase['file'] = output_file
            else:
                print(f"No atoms of phase '{selected_phase['name']}' detected. Skipping export.")
    else:
        print("Error: 'Structure Type' property not found. Cannot export selected phases.")

    try:
        write_phase_manifest(PHASE_MANIFEST, detected_phases, last_computed_data.particles.count,
                             ptm_modifier, input_file, rmsd_cutoff)
        print(f"Phase manifest written to '{PHASE_MANIFEST}'.")
    except (OSError, TypeError, ValueError) as e:
        print(f"Warning: could not write the phase manifest: {e}")

    sys.stdout = logger.terminal
    logger.close()
    
    print(f"\nExport complete. All screen output saved to '{log_filename}'.")

if __name__ == "__main__":
    main()
