"""Generate 648 log-abundance measurements and six binary protein-panel labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from generate_protein_v2 import read_design, generate_interface, LABELS, DEFAULT_SEED

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=60000, help='Multiple of six; at least 12.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, required=True, help='New output NPZ file.')
    args = parser.parse_args()
    if args.n < 12 or args.n % 6:
        parser.error('--n must be a multiple of six and at least 12')
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')
    identity, families, effects = read_design('main')
    primary, y, background, active, sd = generate_interface(args.n, args.seed + 1000, 'multilabel', families, effects)
    x = (background + (active @ effects.T) * sd).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, X=x, Y=y, primary_label=primary,
                            feature_names=np.asarray([row['Aptamer'] for row in identity]),
                            label_names=np.asarray(LABELS))
    print(json.dumps({'generator': 'GPT6-protein', 'synthetic': True, 'seed': args.seed,
                      'shape': list(x.shape), 'positive_counts': y.sum(axis=0).tolist(), 'output': str(args.output)}))

if __name__ == '__main__':
    main()
