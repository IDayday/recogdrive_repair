"""Shared paths for offline analysis; configuration contains paths, never assets."""
from functools import lru_cache
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def config():
    file = Path(os.environ.get('RECOGDRIVE_PATHS', REPO / 'paths.local.json'))
    if not file.is_file():
        raise FileNotFoundError(f'{file}: run discover_local_assets.py first')
    return json.loads(file.read_text())


def path(key):
    value = config().get(key)
    if not value:
        raise ValueError(f'Missing {key} in local path configuration')
    return Path(value).expanduser().resolve()


def native_root():
    return path('native_root')


def code_root():
    return path('code_root')


def logs_root(population):
    return path('navtest_logs' if population == 'test' else 'navtrain_logs')


def sensor_root(population):
    return path('navtest_sensor' if population == 'test' else 'navtrain_sensor')


def analysis_root():
    output = path('analysis_root')
    for key in ('native_root', 'code_root', 'navtrain_logs', 'navtest_logs',
                'navtrain_sensor', 'navtest_sensor'):
        if config().get(key):
            protected = path(key)
            if output == protected or protected in output.parents or output in protected.parents:
                raise ValueError(f'analysis_root overlaps input tree: {protected}')
    if output == REPO or REPO in output.parents or output in REPO.parents:
        raise ValueError('analysis_root must be outside the source repository')
    return output
