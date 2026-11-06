"""Offline descriptive analysis; never loads a model or changes frozen targets."""
from __future__ import annotations
import recogdrive_paths as paths

import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

ROOT = paths.analysis_root()
NATIVE = paths.native_root()
CODE = paths.code_root()
OUT = ROOT / 'scene_metadata'
COMMANDS = ['left', 'straight', 'right', 'stop']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(records)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def read_log(args):
    pop, relative, rows = args
    base = paths.logs_root(pop)
    path = base / relative
    content = path.read_bytes()
    data = pickle.loads(content)
    positions = np.asarray([d['ego2global_translation'][:2] for d in data], np.float64)
    angles = np.asarray([Quaternion(*d['ego2global_rotation']).yaw_pitch_roll[0] for d in data])
    results = []
    for row in rows:
        i = int(row['frame_index'])
        d = data[i]
        assert d['token'] == row['token']
        assert i >= 3 and i + 8 < len(data)
        yaw = angles[i]
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        xy = (positions[i+1:i+9] - positions[i]) @ rot.T
        gt = np.column_stack([xy, wrap(angles[i+1:i+9] - yaw)])
        history_xy = (positions[i-3:i+1] - positions[i]) @ rot.T
        dynamics = np.asarray(d['ego_dynamic_state'], np.float64)
        speed = float(np.linalg.norm(dynamics[:2]))
        boxes = np.asarray(d['anns']['gt_boxes'])
        names = np.asarray(d['anns']['gt_names']).astype(str)
        distance = np.linalg.norm(boxes[:, :2], axis=1)
        vehicle = names == 'vehicle'
        ped = names == 'pedestrian'
        bike = np.isin(names, ['bicycle', 'cyclist'])
        front = (boxes[:, 0] > 0) & (boxes[:, 0] < 30) & (np.abs(boxes[:, 1]) < 2.5)
        velocities = np.asarray(d['anns']['gt_velocity_3d'])
        obj_speed = np.linalg.norm(velocities[:, :2], axis=1)
        delta_gt = np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0)
        gt_path = float(np.linalg.norm(delta_gt, axis=1).sum())
        gt_turn = float(np.degrees(gt[-1, 2]))
        cmd = int(np.argmax(d['driving_command']))
        motion = 'stationary' if gt_path < 2 else ('left' if gt_turn > 15 else ('right' if gt_turn < -15 else 'straight'))
        rec = dict(token=d['token'], global_index=int(row['index']), log_id=row['session'],
                   log_name=row['log_name'], frame_index=i, map_location=d['map_location'],
                   command=COMMANDS[cmd], command_id=cmd, ego_speed_mps=speed,
                   ego_accel_mps2=float(np.linalg.norm(dynamics[2:])),
                   past_turn_deg=float(np.degrees(wrap(angles[i]-angles[i-3]))),
                   past_path_m=float(np.linalg.norm(np.diff(history_xy, axis=0), axis=1).sum()),
                   gt_path_m=gt_path, gt_end_x_m=float(xy[-1, 0]), gt_end_y_m=float(xy[-1, 1]),
                   gt_turn_deg=gt_turn, gt_motion=motion,
                   gt_start=bool(speed < .5 and gt_path > 5),
                   gt_brake_to_stop=bool(speed > 2 and np.linalg.norm(data[i+8]['ego_dynamic_state'][:2]) < 1),
                   vehicles20=int(np.sum(vehicle & (distance < 20))),
                   vehicles40=int(np.sum(vehicle & (distance < 40))),
                   pedestrians20=int(np.sum(ped & (distance < 20))),
                   bicycles20=int(np.sum(bike & (distance < 20))),
                   objects20=int(np.sum(distance < 20)),
                   any_front_object30=bool(np.any(front)),
                   any_front_vehicle30=bool(np.any(front & vehicle)),
                   front_stopped_vehicle20=bool(np.any(front & vehicle & (distance < 20) & (obj_speed < 1))),
                   front_vehicle_min_x_m=float(np.min(boxes[front & vehicle, 0])) if np.any(front & vehicle) else 30.,
                   red_light_observed=bool(any(bool(t[1]) for t in d['traffic_lights'])),
                   lights_observed=len(d['traffic_lights']), route_roadblocks=len(d['roadblock_ids']),
                   future_dt_max_error_s=float(np.max(np.abs((np.asarray([dd['timestamp'] for dd in data[i+1:i+9]])-d['timestamp'])/1e6 - np.arange(1, 9)*.5))))
        results.append((rec, gt.astype(np.float32)))
    return results, dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), scenes=len(rows), bytes=len(content))


def contingency(labels, values, keep=None):
    if keep is None:
        keep = np.ones(len(labels), bool)
    y = np.asarray(labels)[keep]
    v = np.asarray(values).astype(str)[keep]
    keys, vi = np.unique(v, return_inverse=True)
    a = np.zeros((9, len(keys)), np.int64)
    np.add.at(a, (y, vi), 1)
    a = a[a.sum(1) > 0]
    n = a.sum()
    p = a / n
    r, c = p.sum(1), p.sum(0)
    expected = r[:, None] * c[None, :]
    nonzero = p > 0
    mi = float(np.sum(p[nonzero] * np.log(p[nonzero] / expected[nonzero])))
    hy = float(-np.sum(r[r > 0] * np.log(r[r > 0])))
    hv = float(-np.sum(c[c > 0] * np.log(c[c > 0])))
    chi = float(n * np.sum((p-expected)**2 / np.maximum(expected, 1e-100)))
    vstat = float(np.sqrt(chi / n / max(1, min(a.shape)-1)))
    return dict(n=int(n), MI_nats=mi, fraction_choice_entropy_explained=mi/hy if hy else 0,
                symmetric_NMI=2*mi/(hy+hv) if hy+hv else 0, Cramers_V=vstat,
                levels=keys.tolist())


def eta_squared(labels, x, keep):
    labels, x = np.asarray(labels)[keep], np.asarray(x, float)[keep]
    total = np.sum((x-x.mean())**2)
    between = sum(np.sum(labels == j)*(x[labels == j].mean()-x.mean())**2 for j in np.unique(labels))
    return float(between/total) if total else 0.


def bootstrap_diff(group, variable, logs, rng, repeats=1000):
    ids, inv = np.unique(logs, return_inverse=True)
    ng = np.bincount(inv, weights=group, minlength=len(ids))
    no = np.bincount(inv, weights=~group, minlength=len(ids))
    sg = np.bincount(inv, weights=group*variable, minlength=len(ids))
    so = np.bincount(inv, weights=(~group)*variable, minlength=len(ids))
    draws = rng.integers(0, len(ids), (repeats, len(ids)))
    g, o = ng[draws].sum(1), no[draws].sum(1)
    valid = (g > 0) & (o > 0)
    vals = sg[draws].sum(1)[valid]/g[valid] - so[draws].sum(1)[valid]/o[valid]
    return np.quantile(vals, [.025, .975]).tolist() if len(vals) else [None, None]


def analyze(pop):
    with np.load(NATIVE/'frozen_teacher/selections'/f'{pop}.npz', allow_pickle=False) as z:
        tokens, choices, q = z['tokens'].astype(str), z['choice'], z['q_points']
    path = OUT/f'{pop}_metadata.npz'
    if not path.exists():
        with gzip.open(CODE/'manifests'/f'{pop}.csv.gz', 'rt') as f:
            source = {r['token']: r for r in csv.DictReader(f)}
        assert set(source) == set(tokens)
        grouped = defaultdict(list)
        for token in tokens:
            grouped[source[token]['raw_rel']].append(source[token])
        records, gt, files = {}, {}, []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result, ref in pool.map(read_log, [(pop, k, v) for k, v in sorted(grouped.items())]):
                files.append(ref)
                for rec, traj in result:
                    records[rec['token']], gt[rec['token']] = rec, traj
        recs = [records[t] for t in tokens]
        arrays = {k: np.asarray([r[k] for r in recs]) for k in recs[0]}
        np.savez_compressed(path, **arrays, gt_trajectories=np.stack([gt[t] for t in tokens]))
        csv_write(OUT/f'{pop}_metadata.csv', recs)
        save_json(OUT/f'{pop}_raw_sources.json', files)
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    assert np.array_equal(m['token'], tokens)
    assert np.array_equal(q.argmax(1), choices)
    m['speed_bin'] = np.asarray(['<0.5', '0.5-2', '2-5', '5-10', '>=10'])[np.digitize(m['ego_speed_mps'], [.5, 2, 5, 10])]
    m['density_bin'] = np.asarray(['0', '1-3', '4-8', '>=9'])[np.digitize(m['vehicles20'], [1, 4, 9])]
    m['Base_quality_bin'] = np.asarray(['<70', '70-90', '90-99', '99-100', '100'])[np.digitize(q[:, 0], [70, 90, 99, 100])]
    gap = np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2]
    filters = {'all': np.ones(len(tokens), bool), 'repair_chosen': choices > 0,
               'decisive_gap_gt1': gap > 1}
    cats = ['command', 'gt_motion', 'speed_bin', 'map_location', 'density_bin', 'red_light_observed',
            'gt_start', 'gt_brake_to_stop', 'front_stopped_vehicle20', 'Base_quality_bin']
    associations = []
    for maskname, mask in filters.items():
        for k in cats:
            associations.append(dict(population=pop, subset=maskname, variable=k, **contingency(choices, m[k], mask)))
    csv_write(OUT/f'{pop}_associations.csv', [{k:v for k,v in r.items() if k!='levels'} for r in associations])
    save_json(OUT/f'{pop}_associations.json', associations)
    profiles = []
    for j in range(9):
        keep = choices == j
        if not keep.any():
            continue
        gain = q[np.arange(len(q)), choices] - q[:, 0]
        row = dict(population=pop, choice=j, count=int(keep.sum()), fraction=float(keep.mean()),
                   Base_PDMS=float(q[keep, 0].mean()), teacher_PDMS=float(q[keep, j].mean()),
                   gain_PDMS=float(gain[keep].mean()), distinct_logs=int(len(np.unique(m['log_id'][keep]))))
        for k in ['ego_speed_mps', 'ego_accel_mps2', 'gt_path_m', 'gt_turn_deg', 'vehicles20', 'pedestrians20',
                  'bicycles20', 'gt_start', 'gt_brake_to_stop', 'any_front_vehicle30', 'front_stopped_vehicle20',
                  'red_light_observed']:
            row[k] = float(m[k][keep].mean())
        for k in ['command', 'gt_motion']:
            for level in np.unique(m[k]):
                row[f'{k}__{level}'] = float(np.mean(m[k][keep] == level))
        profiles.append(row)
    csv_write(OUT/f'{pop}_choice_profiles.csv', profiles)
    table = []
    for k in cats:
        for val in np.unique(m[k]):
            keep = m[k] == val
            fixed = q[keep].mean(0)
            row = dict(population=pop, variable=k, level=str(val), n=int(keep.sum()),
                       distinct_logs=int(len(np.unique(m['log_id'][keep]))),
                       Base_PDMS=float(fixed[0]), oracle_PDMS=float(q[keep].max(1).mean()),
                       oracle_gain=float((q[keep].max(1)-q[keep, 0]).mean()))
            for j in range(9):
                row[f'choice{j}_share'] = float(np.mean(choices[keep] == j))
                row[f'choice{j}_unconditional_PDMS'] = float(fixed[j])
            table.append(row)
    csv_write(OUT/f'{pop}_subgroup_profiles.csv', table)
    return m, choices, q, profiles


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    populations = {pop: analyze(pop) for pop in ['train', 'test']}
    comparisons = []
    # Effect sizes, not p-value sorting: predeclared variables and all nine choices.
    rng = np.random.default_rng(20261103)
    for pop, (m, choice, q, profiles) in populations.items():
        boolean = {'command_left':m['command']=='left', 'command_right':m['command']=='right',
                   'command_stop':m['command']=='stop', 'low_speed_lt2':m['ego_speed_mps']<2,
                   'gt_turn_abs_gt15':np.abs(m['gt_turn_deg'])>15, 'gt_stationary':m['gt_path_m']<2,
                   'pedestrians20_present':m['pedestrians20']>0, 'dense_vehicles20_ge4':m['vehicles20']>=4,
                   'front_stopped_vehicle20':m['front_stopped_vehicle20'], 'gt_start':m['gt_start'],
                   'gt_brake_to_stop':m['gt_brake_to_stop'],
                   'red_light_observed':m['red_light_observed']}
        for j in range(9):
            group = choice == j
            for name, variable in boolean.items():
                a, b = float(variable[group].mean()), float(variable[~group].mean())
                low, high = bootstrap_diff(group, variable, m['log_id'], rng)
                comparisons.append(dict(population=pop, choice=j, variable=name,
                                        n=int(group.sum()), group_rate=a, rest_rate=b,
                                        difference=a-b, lift_vs_rest=a/b if b else None,
                                        log_bootstrap_diff_low=low, log_bootstrap_diff_high=high))
    csv_write(OUT/'CHOICE_SEMANTIC_ENRICHMENTS.csv', comparisons)
    train, test = populations['train'], populations['test']
    replication=[]
    for j in range(9):
        rows=[r for r in comparisons if r['choice']==j]
        a={r['variable']:r for r in rows if r['population']=='train'}
        b={r['variable']:r for r in rows if r['population']=='test'}
        for name in a:
            aa,bb=a[name],b[name]
            same=np.sign(aa['difference'])==np.sign(bb['difference'])
            ci_a=aa['log_bootstrap_diff_low']*aa['log_bootstrap_diff_high']>0
            ci_b=bb['log_bootstrap_diff_low']*bb['log_bootstrap_diff_high']>0
            replication.append(dict(choice=j, variable=name, train_difference=aa['difference'],
                                    test_difference=bb['difference'], same_direction=bool(same),
                                    both_log_CI_exclude0=bool(ci_a and ci_b),
                                    descriptive_replicated=bool(same and ci_a and ci_b)))
    csv_write(OUT/'TRAIN_TEST_SEMANTIC_REPLICATION.csv', replication)
    save_json(OUT/'SUMMARY.json', dict(status='COMPLETE', populations={p:dict(scenes=len(v[1]),
                logs=len(np.unique(v[0]['log_id'])), future_dt_error_max=float(v[0]['future_dt_max_error_s'].max()))
                for p,v in populations.items()},
                predeclared_thresholds=dict(speed_mps=[.5,2,5,10],GT_turn_deg=15,GT_stationary_path_m=2,
                                           GT_start_speed_mps=.5,GT_start_path_m=5,objects_radius_m=20,
                                           front_corridor_x_m=[0,30],front_corridor_abs_y_m=2.5),
                bootstrap=dict(unit='original log/session',repeats=1000,seed=20261103,
                               intervals='pointwise 95%; exploratory multiplicity, not familywise controlled'),
                limitations=['oracle-selected groups are post-outcome descriptive groups, not causal capabilities',
                             'traffic-light presence is not route-relevance or intersection-ground-truth',
                             'front corridor is an ego-fixed geometric proxy, not a lane-following label',
                             'GT future used for diagnosis only, never a deployable input',
                             'Test has been examined previously, cross-split agreement is descriptive replication'],
                sources={str(p):sha(p) for p in [Path(__file__),NATIVE/'frozen_teacher/BUNDLE.json',
                         NATIVE/'frozen_teacher/library.json',CODE/'manifests/train.csv.gz',CODE/'manifests/test.csv.gz',
                         OUT/'train_metadata.npz',OUT/'test_metadata.npz']}))
    print(json.dumps(dict(status='COMPLETE',out=str(OUT),replicated_patterns=sum(r['descriptive_replicated'] for r in replication))))


if __name__ == '__main__':
    main()
