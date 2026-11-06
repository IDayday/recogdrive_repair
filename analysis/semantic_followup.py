"""Robustness checks using all frozen branches on external diagnostic strata."""
from __future__ import annotations
import recogdrive_paths as paths
import csv
import json
from pathlib import Path
import numpy as np
from analyze_scene_metadata import OUT, NATIVE, ROOT, csv_write, save_json, sha


def standardized_contrast(y, feature, strata):
    levels, inv = np.unique(strata, return_inverse=True)
    n = len(levels)
    yes = np.bincount(inv, weights=feature, minlength=n)
    no = np.bincount(inv, weights=~feature, minlength=n)
    a = np.bincount(inv, weights=y*feature, minlength=n)
    b = np.bincount(inv, weights=y*(~feature), minlength=n)
    valid = (yes >= 10) & (no >= 10)
    if not valid.any():
        return dict(common_support_fraction=0., common_cells=0, adjusted_difference=None)
    weights = (yes+no)[valid]
    delta = a[valid]/yes[valid] - b[valid]/no[valid]
    return dict(common_support_fraction=float(weights.sum()/len(y)), common_cells=int(valid.sum()),
                adjusted_difference=float(np.average(delta, weights=weights)))


def bootstrap_groupmean(x, mask, logs, rng, repeats=1000):
    ids, inv = np.unique(logs, return_inverse=True)
    count = np.bincount(inv, weights=mask, minlength=len(ids))
    sums = np.bincount(inv, weights=x*mask, minlength=len(ids))
    draws = rng.integers(0, len(ids), (repeats, len(ids)))
    denom = count[draws].sum(1)
    vals = sums[draws].sum(1)[denom > 0]/denom[denom > 0]
    return np.quantile(vals, [.025, .975]).tolist()


def main():
    rows, cadence, summaries = [], {}, {}
    rng = np.random.default_rng(20261104)
    for pop in ['train', 'test']:
        with np.load(OUT/f'{pop}_metadata.npz', allow_pickle=False) as z:
            m = {k:z[k] for k in z.files}
        with np.load(NATIVE/'frozen_teacher/selections'/f'{pop}.npz', allow_pickle=False) as z:
            q, choice = z['q_points'], z['choice']
            assert np.array_equal(z['tokens'], m['token'])
        speedbin=np.digitize(m['ego_speed_mps'], [.5, 2, 5, 10])
        basebin=np.digitize(q[:, 0], [70, 90, 95, 99, 100])
        strata=np.asarray([f'{city}|{s}|{b}' for city,s,b in zip(m['map_location'], speedbin, basebin)])
        definitions = {'command_right':m['command']=='right', 'command_left':m['command']=='left',
                       'front_stopped_vehicle20':m['front_stopped_vehicle20'],
                       'gt_brake_to_stop':m['gt_brake_to_stop'], 'gt_start':m['gt_start'],
                       'low_speed_lt2':m['ego_speed_mps']<2,
                       'dense_vehicle20_ge4':m['vehicles20']>=4,
                       'pedestrian20_present':m['pedestrians20']>0}
        clean = m['future_dt_max_error_s'] <= .1
        cadence[pop] = dict(total=len(choice), error_gt_100ms=int((~clean).sum()),
                            fraction_error_gt_100ms=float((~clean).mean()),
                            max_error_s=float(m['future_dt_max_error_s'].max()),
                            meaning='Eight subsequent raw frames, matching native NAVSIM frame selection; nominal four seconds, not all exact timestamp intervals')
        for name, mask in definitions.items():
            for j in range(1, 9):
                delta = q[:, j]-q[:, 0]
                advantage=q[:, j]-q[:, 1:].mean(1)
                low, high=bootstrap_groupmean(delta, mask, m['log_id'], rng)
                adj = standardized_contrast(advantage, mask, strata)
                rows.append(dict(population=pop,feature=name,choice=j,n=int(mask.sum()),
                                 fixed_branch_PDMS=float(q[mask,j].mean()),
                                 Base_PDMS=float(q[mask,0].mean()),
                                 fixed_branch_delta=float(delta[mask].mean()),
                                 fixed_branch_delta_log_CI_low=low,fixed_branch_delta_log_CI_high=high,
                                 delta_rest=float(delta[~mask].mean()),
                                 comparative_advantage_in_feature=float(advantage[mask].mean()),
                                 comparative_advantage_outside_feature=float(advantage[~mask].mean()),
                                 **adj))
        summaries[pop] = dict()
        # Repeat diagnostic future-dependent associations excluding irregular timestamps.
        for j in [2, 7]:
            for name in ['gt_brake_to_stop','gt_start']:
                a = definitions[name]
                summaries[pop][f'E{j}_{name}']=dict(
                    all_rate=float(a[choice==j].mean()),
                    clean_cadence_rate=float(a[(choice==j)&clean].mean()),
                    clean_cadence_n=int(np.sum((choice==j)&clean)))
    csv_write(OUT/'EXTERNAL_STRATUM_FIXED_BRANCH_EFFECTS.csv',rows)
    save_json(OUT/'ROBUSTNESS.json', dict(status='COMPLETE',frame_cadence=cadence,
                future_sensitive_profiles=summaries,
                fixed_branch_analysis='Every scene in each externally defined stratum, irrespective of oracle winner',
                adjusted_contrast='Within map_location x ego_speed_bin x Base_PDMS_bin, difference between feature-present and feature-absent mean(q_E - mean(q_E1..E8)), weighted by combined cell sizes; common cells require >=10 scenes each group',
                limitations=['Adjustment is descriptive, cannot control unmeasured road context or establish causality',
                             'Base PDMS is outcome information used only for offline diagnostic adjustment',
                             'These exploratory comparisons were chosen after viewing initial profiles; not preregistered confirmatory hypotheses',
                             'No independent statistical significance is claimed for adjusted contrasts; common-support coverage is reported'],
                sources={str(p):sha(p) for p in [Path(__file__),OUT/'train_metadata.npz',OUT/'test_metadata.npz']}))
    print(json.dumps(dict(status='COMPLETE',out=str(OUT),rows=len(rows),cadence=cadence)))


if __name__ == '__main__':
    main()
