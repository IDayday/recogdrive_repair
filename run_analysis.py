"""Run the original offline statistics in dependency order, with isolated output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
STAGES = {
    'outcomes': ['outcomes/analyze_outcomes.py', 'outcomes/supplement_outcomes.py', 'outcomes/plot_outcomes.py'],
    'mechanisms': ['representation/analyze_mechanisms.py'],
    'shifts': ['representation/analyze_selected_shifts.py'],
    'metadata': ['analyze_scene_metadata.py', 'semantic_followup.py'],
    'semantics': ['representation/analyze_fit_cluster_semantics.py'],
    'trajectories': ['trajectories/analyze_trajectories.py', 'trajectories/summarize_profiles.py',
                     'trajectories/bootstrap_geometry.py'],
    'gt': ['trajectories/gt_alignment.py', 'trajectories/gt_bootstrap.py'],
    'fit-planning': ['trajectories/fit_cluster_planning.py', 'trajectories/fit_cluster_bootstrap.py'],
    'report': ['build_deliverable.py'],
}
# These three stages are independently runnable. Others consume prior outputs.
DEPS = {'semantics': ['mechanisms', 'metadata'], 'gt': ['metadata', 'trajectories'],
        'fit-planning': ['semantics', 'trajectories'],
        'report': ['outcomes', 'shifts', 'fit-planning', 'gt']}


def ordered(stages, with_dependencies):
    selected = set()

    def add(stage):
        if stage in selected:
            return
        if with_dependencies:
            for dep in DEPS.get(stage, []):
                add(dep)
        selected.add(stage)

    for stage in stages:
        add(stage)
    return [s for s in STAGES if s in selected]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed_caches(source, output):
    # Copy independent caches only; never write through a symlink to old evidence.
    names = ['scene_metadata/train_metadata.npz', 'scene_metadata/test_metadata.npz',
             'scene_metadata/train_raw_sources.json', 'scene_metadata/test_raw_sources.json',
             'trajectories/TRAIN_SELECTED_TRAJECTORIES.npz', 'trajectories/TRAIN_EXTRACTION.json']
    for name in names:
        src, dst = source / name, output / name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if digest(src) != digest(dst):
                    raise ValueError(f'Existing cache differs: {dst}')
            else:
                shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'paths.local.json')
    parser.add_argument('--stages', nargs='+', default=['outcomes'], choices=['all', *STAGES])
    parser.add_argument('--with-dependencies', action='store_true', help='Also run prerequisites; may read all raw logs and Train trajectories')
    parser.add_argument('--reuse-derived', type=Path, help='Copy existing metadata and Base/selected Train trajectory caches')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps({'stages': STAGES, 'dependencies': DEPS}, indent=2))
        return
    os.environ['RECOGDRIVE_PATHS'] = str(args.config.resolve())
    import recogdrive_paths as paths
    output = paths.analysis_root()
    stages = ordered(list(STAGES) if 'all' in args.stages else args.stages, args.with_dependencies)
    plan = {'output': str(output), 'stages': stages, 'scripts': [p for s in stages for p in STAGES[s]],
            'reuse_derived': str(args.reuse_derived) if args.reuse_derived else None}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    for folder in ('outcomes', 'representation', 'scene_metadata', 'trajectories', 'figures', 'cases', 'logs'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    if args.reuse_derived:
        seed_caches(args.reuse_derived.resolve(), output)
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    record = {'plan': plan, 'steps': [], 'status': 'RUNNING'}
    receipt = output / 'RUN.json'
    for stage in stages:
        for relative in STAGES[stage]:
            script = REPO / 'analysis' / relative
            logfile = output / 'logs' / (relative.replace('/', '__') + '.log')
            print(f'Running {relative}; log={logfile}', flush=True)
            started = time.monotonic()
            with logfile.open('w') as stream:
                result = subprocess.run([sys.executable, str(script)], cwd=REPO, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            record['steps'].append({'script': relative, 'sha256': digest(script),
                                    'returncode': result.returncode,
                                    'elapsed_seconds': time.monotonic() - started, 'log': str(logfile)})
            record['status'] = 'FAILED' if result.returncode else 'RUNNING'
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            if result.returncode:
                raise SystemExit(f'{relative} failed ({result.returncode}); see {logfile}')
    record['status'] = 'COMPLETE'
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Complete: {receipt}', flush=True)


if __name__ == '__main__':
    main()
