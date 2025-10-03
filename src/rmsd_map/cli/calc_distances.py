#!/usr/bin/env python3

from datetime import datetime
import numpy as np
import argparse
from pathlib import Path

from rmsd_map.mol_io.cor_reader import read_cor_file
from rmsd_map.rmsd import pipelines as ppl

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD distance matrix')
    parser.add_argument('input_file', type=Path, help='Input COR file')
    parser.add_argument('-o', '--output', type=str, default='distance_matrix',
                        help='Output file prefix (default: distance_matrix)')
    parser.add_argument('-b', '--batch-size', type=int, default=200000,
                        help='Batch size for calculations (default: 200000)')
    parser.add_argument('-r', '--report-interval', type=int, default=5,
                        help='Progress reporting interval (default: 5)')
    args = parser.parse_args()

    print(f"{now()}: Reading and preparing data")
    fragments = read_cor_file(str(args.input_file))
    NFrags = len(fragments)

    names = np.array([a.id for a in fragments])
    perms = fragments[0].perms()

    print(f"Set up a calculation with {NFrags} fragments, {perms.shape[1]} atoms and {perms.shape[0]} symmetries")
    print(f"{now()}: Starting distance calculations")

    mat, df = ppl.distance_matrix_fragments(fragments, args.batch_size, args.report_interval )

    print(f"Saving data")
    print(f"{now()}: Saving DataFrame")

    df.write_csv(f"{args.output}.csv")

    print(f"{now()}: Saving matrix")

    np.savez(f"{args.output}.npz", names = names, distances = mat)

    print(f"{now()}: Finished!")

if __name__ == '__main__':
    main()
