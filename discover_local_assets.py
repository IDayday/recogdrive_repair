"""Find local experiment inputs using bounded, shallow filesystem inspection."""
import argparse
import json
import os
from pathlib import Path

EXPERIMENT = 'recogdrive_dc8_native_candidates_20261027'
ARCHIVE = 'recogdrive_repair_pattern_analysis_full.zip'
CHECKPOINT_SHA256 = '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20'


def first(candidates, predicate=lambda p: p.exists()):
    return next((str(p.resolve()) for p in candidates if predicate(p)), '')


def discover(roots):
    roots = [Path(p).expanduser().resolve() for p in roots]
    # Only inspect directory names at the first level. Never recurse into caches.
    children = []
    for root in roots:
        try:
            children.extend(Path(e.path) for e in os.scandir(root) if e.is_dir())
        except OSError:
            continue
    projects = roots + children
    native = first([p / 'experiments' / EXPERIMENT for p in projects] +
                   [p / EXPERIMENT for p in roots] +
                   [p for p in roots if p.name == EXPERIMENT],
                   lambda p: (p / 'frozen_teacher/BUNDLE.json').is_file())
    codes = sorted((p for p in projects if p.name.startswith('recogdrive-no-routing-round1-')), reverse=True)
    code = first(codes, lambda p: (p / 'manifests/train.csv.gz').is_file() and
                 (p / 'interface_teacher/native_cap.py').is_file())
    legacy = [p / 'paths.local.json' for p in projects
              if p.name.startswith('recogdrive-teacher2-a800-')]
    legacy_paths = first(legacy, lambda p: p.is_file())
    previous = json.loads(Path(legacy_paths).read_text()) if legacy_paths else {}
    result = {k: previous.get(k, '') for k in (
        'checkpoint', 'vlm', 'maps', 'navtrain_logs', 'navtest_logs',
        'navtrain_sensor', 'navtest_sensor', 'navtrain_metric', 'navtest_metric')}
    for key, name in [('navtrain_logs', 'trainval_navsim_logs'),
                      ('navtest_logs', 'test_navsim_logs'),
                      ('navtrain_sensor', 'trainval_sensor_blobs'),
                      ('navtest_sensor', 'test_sensor_blobs'), ('maps', 'maps')]:
        if not result[key] or not Path(result[key]).exists():
            result[key] = first([p / name for p in roots])
    if not result['checkpoint']:
        result['checkpoint'] = first([p / 'checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt' for p in projects])
    if not result['vlm']:
        result['vlm'] = first([p / 'checkpoints/recogdrive/ReCogDrive-VLM-2B' for p in projects])
    result.update(native_root=native, code_root=code,
                  analysis_root=str(roots[0] / 'recogdrive-repair-analysis-output'),
                  original_analysis=str(Path(native) / 'engineering/repair_patterns_20261103') if native else '',
                  archive=first([p / 'outputs/recogdrive_repair_pattern_evidence_20261103' / ARCHIVE for p in projects]),
                  expected_checkpoint_sha256=CHECKPOINT_SHA256,
                  model_paths_source=legacy_paths)
    return result


def checks(config):
    native = Path(config['native_root']) if config['native_root'] else None
    required = ['frozen_teacher/BUNDLE.json', 'frozen_teacher/library.json',
                'frozen_teacher/space.npz', 'fitting/ORDINARY_LOCAL.npz',
                'fitting/ALLTRAIN/SPACE_AND_CLUSTER.pt',
                'fitting/INNER/SPACE_AND_CLUSTER.pt', 'fitting/ALLTRAIN/CLUSTER_TRANSFER.pt']
    required += [f'frozen_teacher/slot{i:02d}.npz' for i in range(8)]
    for pop in ('train', 'test'):
        required += [f'requests/{pop}/MATRIX.npz', f'frozen_teacher/selections/{pop}.npz',
                     f'engineering/independent_full_teacher_effects_20261027/{pop.upper()}_SCENES.csv']
    required += ['engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz',
                 'requests/train/REPAIRED_CACHE_INDEX.jsonl']
    status = {str(native / p) if native else p: bool(native and (native / p).is_file()) for p in required}
    code = Path(config['code_root']) if config['code_root'] else None
    for rel in ('manifests/train.csv.gz', 'manifests/test.csv.gz', 'interface_teacher/native_cap.py'):
        status[str(code / rel) if code else rel] = bool(code and (code / rel).is_file())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', default=['/mnt/project', '/mnt/navsim'])
    parser.add_argument('--output', type=Path, default=Path('paths.local.json'))
    args = parser.parse_args()
    config = discover(args.roots)
    if args.output.exists():
        parser.error(f'Refusing to replace existing path configuration: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    status = checks(config)
    print(json.dumps({'config': str(args.output.resolve()), 'paths': config,
                      'required_input_files': status,
                      'missing_input_files': [p for p, ok in status.items() if not ok],
                      'model_paths_exist': {k: bool(config[k] and Path(config[k]).exists())
                                            for k in ('checkpoint', 'vlm')},
                      'note': 'Existence check only; archive, checkpoint and embedded source references are not rehashed.'},
                     ensure_ascii=False, indent=2))
    return 0 if all(status.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
