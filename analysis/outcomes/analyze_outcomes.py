import recogdrive_paths as paths
"""Read existing complete teacher score banks only; no model calls or fitting."""
from pathlib import Path
import json, csv, hashlib, itertools
import numpy as np
N=paths.native_root()
OUT=paths.analysis_root() / 'outcomes'
FIELDS=['PDMS','NC','DAC','TTC','Progress','Comfort','DDC']
BOOT=4000

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def writejson(name,x):
 (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def writecsv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])
def js(p,q):
 p=np.asarray(p,float);q=np.asarray(q,float);p=p/p.sum();q=q/q.sum();m=(p+q)/2
 a=p>0;b=q>0
 return float(.5*np.sum(p[a]*np.log2(p[a]/m[a]))+.5*np.sum(q[b]*np.log2(q[b]/m[b])))
class Boot:
 def __init__(self,logs,seed):
  self.labs,self.inv=np.unique(logs,return_inverse=True);self.L=len(self.labs)
  self.mult=np.random.default_rng(seed).multinomial(self.L,np.ones(self.L)/self.L,size=BOOT)
 def mean(self,v,mask=None):
  v=np.asarray(v,float)
  if mask is None:mask=np.ones(len(v),bool)
  if not np.any(mask):return {'mean':None,'ci95':None,'n':0,'logs':0}
  ss=np.bincount(self.inv,weights=np.where(mask,v,0),minlength=self.L)
  nn=np.bincount(self.inv,weights=mask.astype(float),minlength=self.L)
  num=self.mult@ss;den=self.mult@nn;values=num[den>0]/den[den>0]
  return {'mean':float(v[mask].mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),'n':int(mask.sum()),'logs':int((nn>0).sum())}

def main():
 all_results={};sources=[];branch_rows=[];failure_rows=[];strata_rows=[];ablation_rows=[];noise_rows=[];base_rows=[];component_rows=[]
 profiles={}
 for pi,pop in enumerate(['train','test']):
  mp=N/f'requests/{pop}/MATRIX.npz';sp=N/f'frozen_teacher/selections/{pop}.npz'
  cp=N/f'engineering/independent_full_teacher_effects_20261027/{pop.upper()}_SCENES.csv'
  for p in [mp,sp,cp]:sources.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
  with np.load(mp) as z:s=z['scores'];indices=z['indices']
  with np.load(sp) as z:
   tokens=z['tokens'].astype(str);choice=z['choice'];q=z['q_points'];idx=z['indices']
  assert np.array_equal(indices,idx)
  assert s.shape==(len(tokens),9,5,7)
  assert np.allclose(q,s[:,:,:,0].mean(2)*100,atol=0,rtol=0)
  assert np.array_equal(choice,np.argmax(q,axis=1))
  mapping={}
  with cp.open() as f:
   for row in csv.DictReader(f):mapping[row['token']]=row['log_id']
  assert len(mapping)==len(tokens)
  logs=np.array([mapping[t] for t in tokens]);b=Boot(logs,20261103+pi)
  n=len(tokens);ar=np.arange(n);base=q[:,0];teacher=q[ar,choice];gain=teacher-base
  bs=s[:,0];ts=s[ar,choice];delta=ts-bs
  top2=np.sort(q,axis=1)[:,-2:];margin=top2[:,1]-top2[:,0];near=(q>=teacher[:,None]-1.)
  stable=np.argmax(s[:,:,:,0],axis=1)
  exact_best=(s[:,:,:,0]==s[:,:,:,0].max(1)[:,None,:])
  chosen_is_best=exact_best[ar,choice]
  agreement=(stable==choice[:,None]).mean(1)
  component_mean=delta.mean(1)
  r={'scenes':n,'logs':b.L,'base':float(base.mean()),'teacher':float(teacher.mean()),'gain':b.mean(gain),
    'choice_counts':np.bincount(choice,minlength=9).tolist(),'all_noise_stable_argmax_same_fraction':float(np.all(stable==choice[:,None],1).mean()),
    'chosen_exactly_noise_optimal_all5_fraction':float(chosen_is_best.all(1).mean()),
    'chosen_exactly_noise_optimal_noise_fraction':float(chosen_is_best.mean()),
    'winner_gap_gt1_scene_fraction':float((margin>1).mean()),
    'winner_gap_gt1_repair_fraction':float((margin[choice>0]>1).mean()),
    'winner_gap_gt1_stable_argmax_all5_fraction':float(np.all(stable[margin>1]==choice[margin>1,None],1).mean()),
    'near_optimal_1point_count_mean':float(near.sum(1).mean()),
    'single_noise_oracle_PDMS':float(s[:,:,:,0].max(1).mean()*100),
    'single_noise_oracle_advantage_over_fixed':float(s[:,:,:,0].max(1).mean()*100-teacher.mean()),
    'per_seed_teacher_vs_base_mean_points':(delta[:,:,0].mean(0)*100).tolist(),
    'scene_noise_negative_gain_fraction':float((delta[:,:,0]<-1e-12).mean()),
    'teacher_any_noise_below_base_scene_fraction':float((delta[:,:,0]<-1e-12).any(1).mean()),
    'teacher_mean_gain_gt1_fraction':float((gain>1).mean()),'teacher_zero_recoveries_noise_count':int(((bs[:,:,0]==0)&(ts[:,:,0]>0)).sum()),
    'teacher_new_zero_noise_count':int(((bs[:,:,0]>0)&(ts[:,:,0]==0)).sum()),
    'metric_mean_Base':bs.mean((0,1)).tolist(),'metric_mean_teacher':ts.mean((0,1)).tolist()}
  for c in range(9):
   mask=choice==c;uncond=q[:,c]-base
   ci=b.mean(gain,mask)
   unique=margin>1
   without=q.copy();without[:,c]=-np.inf
   deletion=teacher-np.max(without,axis=1)
   br={'population':pop,'choice':c,'slot':c-1 if c else 'Base','n':int(mask.sum()),'scene_pct':float(mask.mean()*100),
      'selected_base_PDMS':float(base[mask].mean()) if mask.any() else None,
      'selected_teacher_PDMS':float(teacher[mask].mean()) if mask.any() else None,
      'selected_gain':ci['mean'],'selected_gain_ci_low':ci['ci95'][0] if ci['ci95'] else None,'selected_gain_ci_high':ci['ci95'][1] if ci['ci95'] else None,
      'whole_population_gain_contribution':float(gain[mask].sum()/n),
      'fraction_total_gain_pct':float(gain[mask].sum()/gain.sum()*100),
      'all_scenes_fixed_branch_PDMS':float(q[:,c].mean()),'all_scenes_fixed_branch_delta':float(uncond.mean()),
      'all_scenes_fixed_branch_worse_than_Base_pct':float((uncond<0).mean()*100),
      'all_scenes_fixed_branch_gt1_gain_pct':float((uncond>1).mean()*100),
      'base_plus_this_branch_oracle_PDMS':float(np.maximum(base,q[:,c]).mean()),
      'delete_this_branch_loss_PDMS':float(deletion.mean()),
      'winner_margin_gt1_pct_among_selected':float(unique[mask].mean()*100) if mask.any() else None,
      'noise_exact_best_all5_pct_among_selected':float(chosen_is_best[mask].all(1).mean()*100) if mask.any() else None,
      'noise_stable_argmax_all5_pct_among_selected':float((stable[mask]==choice[mask,None]).all(1).mean()*100) if mask.any() else None}
   for k in range(1,7):br[f'selected_delta_{FIELDS[k]}_pp']=float(component_mean[mask,k].mean()*100) if mask.any() else None
   branch_rows.append(br)
   noise_rows.append({'population':pop,'choice':c,'n':int(mask.sum()),'mean_5noise_argmax_agreement':float(agreement[mask].mean()) if mask.any() else None,
       'mean_nearopt1pt_candidate_count':float(near[mask].sum(1).mean()) if mask.any() else None})
  for k in range(1,7):
   ci=b.mean(component_mean[:,k]*100)
   component_rows.append({'population':pop,'metric':FIELDS[k],'base':float(bs[:,:,k].mean()),'teacher':float(ts[:,:,k].mean()),'delta_pp':ci['mean'],'ci95_low':ci['ci95'][0],'ci95_high':ci['ci95'][1]})
  for c in range(9):
   cm=choice==c
   for k in [1,2,3,5,6]:
    bf=bs[:,:,k]<1.;tf=ts[:,:,k]<1.
    fail=bf[cm];repaired=(bf&~tf)[cm];new=(~bf&tf)[cm]
    failure_rows.append({'population':pop,'choice':c,'metric':FIELDS[k], 'selected_scenes':int(cm.sum()),
      'base_failed_scene_noises':int(fail.sum()),'fully_repaired_scene_noises':int(repaired.sum()),
      'recovery_pct_of_base_failed':float(repaired.sum()/fail.sum()*100) if fail.sum() else None,
      'new_failed_scene_noises':int(new.sum()),'teacher_failed_scene_noises':int(tf[cm].sum()),
      'base_failure_pct':float(fail.mean()*100) if fail.size else None,
      'teacher_failure_pct':float(tf[cm].mean()*100) if tf[cm].size else None})
  bounds=[(0,70),(70,90),(90,95),(95,99),(99,100.0000000001)]
  for lo,hi in bounds:
   mask=(base>=lo)&(base<hi);ci=b.mean(gain,mask)
   row={'population':pop,'stratum':f'[{lo},{hi})' if hi<=100 else '[99,100]','n':int(mask.sum()),'scene_pct':float(mask.mean()*100),
     'Base_PDMS':float(base[mask].mean()),'teacher_PDMS':float(teacher[mask].mean()),'gain':ci['mean'],'gain_ci_low':ci['ci95'][0],'gain_ci_high':ci['ci95'][1],
     'total_gain_contribution':float(gain[mask].sum()/n),'gain1_pct':float((gain[mask]>1).mean()*100),
     'Base_selected_pct':float((choice[mask]==0).mean()*100)}
   for c in range(1,9):row[f'choice{c}_pct']=float((choice[mask]==c).mean()*100)
   strata_rows.append(row)
  fmasks={'all_safety_all5_pass':np.all(bs[:,:,[1,2,3]]==1,axis=(1,2)),
   'any_NC_failure':(bs[:,:,1]<1).any(1),'any_DAC_failure':(bs[:,:,2]<1).any(1),'any_TTC_failure':(bs[:,:,3]<1).any(1),
   'any_safety_failure':(bs[:,:,[1,2,3]]<1).any((1,2)),
   'all_safety_pass_progress_lt1':np.all(bs[:,:,[1,2,3]]==1,axis=(1,2))&(bs[:,:,4].mean(1)<1),
   'all5_PDMS100':np.all(bs[:,:,0]==1,1)}
  for name,mask in fmasks.items():
   ci=b.mean(gain,mask)
   base_rows.append({'population':pop,'baseline_stratum':name,'n':int(mask.sum()),'scene_pct':float(mask.mean()*100),
    'Base_PDMS':float(base[mask].mean()) if mask.any() else None,'teacher_PDMS':float(teacher[mask].mean()) if mask.any() else None,
    'gain':ci['mean'],'ci_low':ci['ci95'][0] if ci['ci95'] else None,'ci_high':ci['ci95'][1] if ci['ci95'] else None,
    'total_gain_contribution':float(gain[mask].sum()/n),'fraction_total_gain':float(gain[mask].sum()/gain.sum()),
    'gain_gt1_fraction':float((gain[mask]>1).mean()) if mask.any() else None})
  frontier={}
  for k in range(9):
   best=-1;bestset=()
   for subset in itertools.combinations(range(1,9),k):
    value=float(q[:,(0,)+subset].max(1).mean())
    if value>best:best=value;bestset=subset
   frontier[str(k)]={'choices':list(bestset),'oracle_PDMS':best,'gain_retained_pct':float((best-base.mean())/(teacher.mean()-base.mean())*100)}
  r['within_split_oracle_subset_frontier']=frontier
  r['margin_gt1_gain']=b.mean(gain,margin>1)
  r['base_mean_zero_scenes']=int((base==0).sum());r['teacher_mean_zero_scenes']=int((teacher==0).sum())
  r['full_score100_base_scene_count']=int((base==100).sum())
  all_results[pop]=r
  profiles[pop]={'q':q,'gain':gain,'choice':choice,'logs':logs}
  np.savez(OUT/f'{pop.upper()}_OUTCOME_FEATURES.npz',tokens=tokens,indices=indices,logs=logs,choice=choice,base=base,teacher=teacher,gain=gain,
   winner_margin=margin,nearopt1pt_count=near.sum(1),noise_argmax_agreement=agreement,all5_chosen_exact_best=chosen_is_best.all(1),base_metrics=bs.mean(1),teacher_metrics=ts.mean(1))
 for k,f in all_results['train']['within_split_oracle_subset_frontier'].items():
  subset=(0,)+tuple(f['choices']);test_q=profiles['test']['q'];val=float(test_q[:,subset].max(1).mean())
  ablation_rows.append({'repair_choices_k':int(k),'Train_picked_choices':','.join(map(str,f['choices'])),'Train_oracle_PDMS':f['oracle_PDMS'],
   'Train_gain_retained_pct':f['gain_retained_pct'],'Test_oracle_PDMS_same_subset':val,
   'Test_gain_retained_pct_same_subset':float((val-all_results['test']['base'])/all_results['test']['gain']['mean']*100),
   'Test_hindsight_optimal_subset_PDMS':all_results['test']['within_split_oracle_subset_frontier'][k]['oracle_PDMS']})
 tr=[r for r in branch_rows if r['population']=='train' and r['choice']>0];te=[r for r in branch_rows if r['population']=='test' and r['choice']>0]
 cross={'branch_usage_pearson':corr([r['scene_pct'] for r in tr],[r['scene_pct'] for r in te]),
   'branch_selected_gain_pearson':corr([r['selected_gain'] for r in tr],[r['selected_gain'] for r in te]),
   'branch_unconditional_delta_pearson':corr([r['all_scenes_fixed_branch_delta'] for r in tr],[r['all_scenes_fixed_branch_delta'] for r in te]),
   'branch_gain_contribution_pearson':corr([r['whole_population_gain_contribution'] for r in tr],[r['whole_population_gain_contribution'] for r in te]),
   'choice_distribution_Jensen_Shannon_bits':js(all_results['train']['choice_counts'],all_results['test']['choice_counts']),
   'component_profile_by_choice':[]}
 for rt,re in zip(tr,te):
  cols=[f'selected_delta_{f}_pp' for f in FIELDS[1:]]
  cross['component_profile_by_choice'].append({'choice':rt['choice'],'all6metric_delta_pearson':corr([rt[x] for x in cols],[re[x] for x in cols]),
   'NC_DAC_TTC_Progress_delta_pearson':corr([rt[x] for x in cols[:4]],[re[x] for x in cols[:4]])})
 writecsv('BRANCH_PROFILES.csv',branch_rows);writecsv('SAFETY_RECOVERY_BY_BRANCH.csv',failure_rows);writecsv('BASE_SCORE_STRATA.csv',strata_rows)
 writecsv('BASE_FAILURE_STRATA.csv',base_rows);writecsv('TRAIN_PICKED_SUBSET_FRONTIER.csv',ablation_rows)
 writecsv('NOISE_STABILITY_BY_BRANCH.csv',noise_rows);writecsv('COMPONENT_DELTA.csv',component_rows)
 writejson('SUMMARY.json',all_results);writejson('TRAIN_TEST_SIMILARITY.json',cross)
 sources.append({'path':str(Path(__file__).resolve()),'sha256':digest(__file__)})
 writejson('PROVENANCE.json',{'sources':sources,'inference':False,'training':False,'score_recomputation':False,
  'bootstrap':{'type':'paired whole-log cluster bootstrap; natural scene-equal ratio of sums','draws':BOOT,'seeds':[20261103,20261104],
    'limits':'CIs condition on fixed trained model, fixed candidate library and evaluation noise bank; no between-training-seed variation.'},
  'selection':'actual frozen Base-first exact argmax of five-noise mean; Base+expert subset oracles diagnostic, not deployable selectors',
  'conditional_profiles':'Grouping by post-hoc winner necessarily favors that winner; do not infer causal semantic classes or router generalization.'})
 print(json.dumps({'train':{k:all_results['train'][k] for k in ['base','teacher','gain','winner_gap_gt1_scene_fraction','near_optimal_1point_count_mean']},
 'test':{k:all_results['test'][k] for k in ['base','teacher','gain','winner_gap_gt1_scene_fraction','near_optimal_1point_count_mean']},'cross':cross},indent=2))
if __name__=='__main__':main()
