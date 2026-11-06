import recogdrive_paths as paths
from pathlib import Path
import csv,hashlib,json
import numpy as np
from analyze_trajectories import features,sha,csvout,js
O=paths.analysis_root() / 'trajectories';N=paths.native_root()

def main():
 out=[];scene_rows=[];provenance=[]
 for pop in ['train','test']:
  meta_path=O.parent/'scene_metadata'/f'{pop}_metadata.npz'
  with np.load(meta_path,allow_pickle=False) as z:meta={k:z[k] for k in z.files}
  if pop=='train':
   p=O/'TRAIN_SELECTED_TRAJECTORIES.npz'
   with np.load(p,allow_pickle=False) as z:tokens=z['tokens'];choice=z['choice'];t=z['trajectories'].astype(np.float64)
  else:
   p=N/'engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz'
   with np.load(p,allow_pickle=False) as z:tokens=z['tokens'];tr=z['trajectories']
   with np.load(N/'frozen_teacher/selections/test.npz',allow_pickle=False) as z:choice=z['choice']
   t=np.stack([tr[:,0],tr[np.arange(len(choice)),choice]],axis=1).astype(np.float64)
  assert np.array_equal(meta['token'],tokens)
  gt=meta['gt_trajectories'].astype(np.float64);delta=t[...,:2]-gt[:,None,None,:,:2]
  euclidean=np.linalg.norm(delta,axis=-1);tan=np.stack([np.cos(gt[...,2]),np.sin(gt[...,2])],axis=-1)
  along=(delta*tan[:,None,None]).sum(-1);lat=delta[...,0]*(-tan[:,None,None,:,1])+delta[...,1]*tan[:,None,None,:,0]
  paths=features(t)['path_m'];patherror=paths-meta['gt_path_m'][:,None,None]
  arrays={'GT_ADE_m':euclidean.mean((2,3)),'GT_FDE_m':euclidean[:,:,:,-1].mean(2),'GT_mean_alongtrack_error_m':along.mean((2,3)),'GT_endpoint_alongtrack_error_m':along[:,:,:,-1].mean(2),'GT_mean_abs_alongtrack_error_m':np.abs(along).mean((2,3)),'GT_mean_abs_crosstrack_error_m':np.abs(lat).mean((2,3)),'GT_end_x_error_m':delta[:,:,:,-1,0].mean(2),'GT_end_y_error_m':delta[:,:,:,-1,1].mean(2),'GT_path_error_m':patherror.mean(2),'GT_abs_path_error_m':np.abs(patherror).mean(2)}
  with np.load(N/f'frozen_teacher/selections/{pop}.npz',allow_pickle=False) as z:q=z['q_points'];assert np.array_equal(z['tokens'],tokens)
  gain=q[np.arange(len(choice)),choice]-q[:,0]
  timing_ok=meta['future_dt_max_error_s']<=.1
  for sample_name,valid in [('all_scenes',np.ones(len(choice),dtype=bool)),('nominal_dt_error_le0p1s',timing_ok),('GT_start',meta['gt_start']),('GT_brake_to_stop',meta['gt_brake_to_stop']),('front_stopped_vehicle20',meta['front_stopped_vehicle20'])]:
   for c in [-2,-1,*range(9)]:
    m=valid&((choice>=0) if c==-2 else (choice>0) if c==-1 else (choice==c))
    if not m.any():continue
    r=dict(population=pop,sample=sample_name,choice=c,slot='all' if c==-2 else 'all_repaired' if c==-1 else c-1,n=int(m.sum()),gain_points_mean=float(gain[m].mean()),ego_speed_mps=float(meta['ego_speed_mps'][m].mean()),gt_path_m=float(meta['gt_path_m'][m].mean()),gt_start_pct=float(meta['gt_start'][m].mean()*100),gt_brake_to_stop_pct=float(meta['gt_brake_to_stop'][m].mean()*100),front_stopped_vehicle20_pct=float(meta['front_stopped_vehicle20'][m].mean()*100),future_dt_error_gt0p1s_pct=float((~timing_ok)[m].mean()*100))
    for k,a in arrays.items():r['Base_'+k]=float(a[m,0].mean());r['teacher_'+k]=float(a[m,1].mean());r['delta_'+k]=float((a[m,1]-a[m,0]).mean())
    r['GT_ADE_improved_pct']=float((arrays['GT_ADE_m'][m,1]<arrays['GT_ADE_m'][m,0]-1e-8).mean()*100)
    r['positive_PDMS_but_GT_ADE_worse_pct']=float(((gain[m]>1e-8)&(arrays['GT_ADE_m'][m,1]>arrays['GT_ADE_m'][m,0]+1e-8)).mean()*100)
    out.append(r)
  rows=[]
  for i in range(len(choice)):
   r=dict(population=pop,token=str(tokens[i]),choice=int(choice[i]),future_dt_max_error_s=float(meta['future_dt_max_error_s'][i]),gt_start=int(meta['gt_start'][i]),gt_brake_to_stop=int(meta['gt_brake_to_stop'][i]),gain_points=float(gain[i]))
   for k,a in arrays.items():r['Base_'+k]=float(a[i,0]);r['teacher_'+k]=float(a[i,1]);r['delta_'+k]=float(a[i,1]-a[i,0])
   rows.append(r)
  csvout(O/f'{pop.upper()}_GT_ALIGNMENT_PER_SCENE.csv',rows)
  provenance.append(dict(population=pop,scenes=len(choice),metadata=str(meta_path),metadata_sha256=sha(meta_path),trajectories_source=str(p),source_sha256=sha(p),timing_valid_n=int(timing_ok.sum()),timing_invalid_n=int((~timing_ok).sum())))
 csvout(O/'GT_ALIGNMENT_BY_CHOICE_AND_CONDITION.csv',out)
 js(O/'GT_ALIGNMENT_SUMMARY.json',dict(rows=out,provenance=provenance,meaning='Actual recorded GT is diagnostic, not a universal optimal PDMS target. Raw NAVSIM eight-future-frame GT is matched as dataset implementation; nominal 0.5sec index may have occasional timestamp gaps. Report all plus <=0.1sec timing-error sensitivity. Signed alongtrack error uses each GT heading direction, not global x.'))
 for r in out:
  if r['sample']=='all_scenes' and r['choice'] in [-1,2,6,7,8]:print(json.dumps(r))

if __name__=='__main__':main()
