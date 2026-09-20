"""Generate FIRST regional volumes for the supported AD/MCI and SCZ tasks."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def load_engine(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ad-binary', 'ad-multiclass', 'scz'], required=True)
    parser.add_argument('--n-per-class', type=int, default=1000)
    parser.add_argument('--seed', type=int, help='Defaults: AD 20260908; SCZ 20260918.')
    parser.add_argument('--output', type=Path, required=True, help='New output NPZ file.')
    args = parser.parse_args()
    if args.n_per_class < 2:
        parser.error('--n-per-class must be at least 2')
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')
    if args.task.startswith('ad-'):
        seed = 20260908 if args.seed is None else args.seed
        engine = load_engine('ad_engine', ROOT / 'ad/generate_ad_v2.py')
        arrays, y, classes, names, state = engine.make('first', args.task[3:], args.n_per_class, seed)
        x = arrays['knowledge']
    else:
        seed = 20260918 if args.seed is None else args.seed
        engine = load_engine('scz_engine', ROOT / 'scz/generate_scz_v2.py')
        engine.SEED = seed
        base, amplitude, permutation, effects, sigma, names = engine.background(engine.spec('first'), args.n_per_class)
        x = engine.arm_array(base, amplitude, permutation, effects, sigma, 'knowledge')
        y = np.repeat(np.arange(2), args.n_per_class)
        classes = engine.CLASSES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, X=x, y=y, feature_names=np.asarray(names), class_names=np.asarray(classes))
    print(json.dumps({'generator': 'GPT6-Brain', 'task': args.task, 'synthetic': True,
                      'seed': seed, 'shape': list(x.shape), 'output': str(args.output)}))

if __name__ == '__main__':
    main()
