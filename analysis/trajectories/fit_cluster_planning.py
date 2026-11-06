import recogdrive_paths as paths
from pathlib import Path
import csv,json
import numpy as np
from analyze_trajectories import features,csvout,js,sha
O=paths.analysis_root() / 'trajectories';N=paths.native_root()

def main():
 fp=O.parent/'representation/FIT_PAIR_CLUSTERS.csv'
 with fp.open() as f:fit=list(csv.DictReader(f))
 mp=O.parent/'scene_metadata/train_metadata.npz'
 with np.load(mp,allow_pickle=False) as z:meta={k:z[k] for k in z.files}
 tp=O/'TRAIN_SELECTED_TRAJECTORIES.npz'
 with np.load(tp,allow_pickle=False) as z:
  tok=z['tokens'];assert np.array_equal(tok,meta['token']);base=z['trajectories'][:,0].astype(np.float64)
 lookup={str(t):i for i,t in enumerate(tok)};ix=np.asarray([lookup[r['token']] for r in fit]);cl=np.asarray([int(r['fit_cluster']) for r in fit])
 b=base[ix];gt=meta['gt_trajectories'][ix].astype(np.float64);delta=b[...,:2]-gt[:,None,:,:2]
 tang=np.stack([np.cos(gt[...,2]),np.sin(gt[...,2])],-1)
 along=(delta*tang[:,None]).sum(-1);cross=(delta*np.stack([-tang[...,1],tang[...,0]],-1)[:,None]).sum(-1)
 head=(b[...,2]-gt[:,None,:,2]+np.pi)%(2*np.pi)-np.pi
 eu=np.linalg.norm(delta,-1) if False else np.linalg.norm(delta,axis=-1)
 feat=features(b)
 arr={'GT_ADE_m':eu.mean(-1),'GT_FDE_m':eu[...,-1],'GT_end_x_error_m':delta[...,-1,0],'GT_end_y_error_m':delta[...,-1,1],'GT_endpoint_alongtrack_error_m':along[...,-1],'GT_endpoint_crosstrack_error_m':cross[...,-1],'GT_mean_alongtrack_error_m':along.mean(-1),'GT_mean_abs_crosstrack_error_m':np.abs(cross).mean(-1),'GT_final_heading_error_deg':np.rad2deg(head[...,-1]),'GT_final_abs_heading_error_deg':np.abs(np.rad2deg(head[...,-1])),'GT_path_error_m':feat['path_m']-meta['gt_path_m'][ix,None],'Base_path_m':feat['path_m'],'Base_final_heading_deg':feat['final_heading_deg']}
 with np.load(N/'requests/train/MATRIX.npz',allow_pickle=False) as z:sc=z['scores'][ix,0]
 rec=[];rows=[]
 for agg,noise in [('S0',0),('S0S4_mean',None)]:
  a={k:v[:,noise] if noise is not None else v.mean(1) for k,v in arr.items()}
  score=sc[:,noise,0]*100 if noise is not None else sc[:,:,0].mean(1)*100
  for condition,valid in [('all_fit_pairs',np.ones(len(ix),bool)),('GT_start',meta['gt_start'][ix]),('GT_brake_to_stop',meta['gt_brake_to_stop'][ix]),('GT_left_turn',meta['gt_turn_deg'][ix]>20),('GT_right_turn',meta['gt_turn_deg'][ix]<-20),('nominal_dt_error_le0p1s',meta['future_dt_max_error_s'][ix]<=.1)]:
   for c in range(8):
    m=(cl==c)&valid
    if not m.any():continue
    r=dict(fit_cluster=c,noise_aggregation=agg,condition=condition,n=int(m.sum()),Base_PDMS=float(score[m].mean()),ego_speed_mps=float(meta['ego_speed_mps'][ix][m].mean()),gt_path_m=float(meta['gt_path_m'][ix][m].mean()),gt_end_x_m=float(meta['gt_end_x_m'][ix][m].mean()),gt_turn_deg=float(meta['gt_turn_deg'][ix][m].mean()),gt_start_pct=float(meta['gt_start'][ix][m].mean()*100),gt_brake_to_stop_pct=float(meta['gt_brake_to_stop'][ix][m].mean()*100),right_command_pct=float((meta['command'][ix][m]=='right').mean()*100),left_command_pct=float((meta['command'][ix][m]=='left').mean()*100))
    r.update({k:float(v[m].mean()) for k,v in a.items()});r['ahead_GT_end_gt0p5m_pct']=float((a['GT_endpoint_alongtrack_error_m'][m]>.5).mean()*100);r['behind_GT_end_ltminus0p5m_pct']=float((a['GT_endpoint_alongtrack_error_m'][m]<-.5).mean()*100)
    rows.append(r)
  if agg=='S0S4_mean':
   for j in range(len(ix)):
    r=dict(token=fit[j]['token'],fit_cluster=int(cl[j]),global_index=int(meta['global_index'][ix[j]]),ego_speed_mps=float(meta['ego_speed_mps'][ix[j]]),gt_start=int(meta['gt_start'][ix[j]]),gt_brake_to_stop=int(meta['gt_brake_to_stop'][ix[j]]),gt_turn_deg=float(meta['gt_turn_deg'][ix[j]]),Base_PDMS=float(score[j]))
    r.update({k:float(v[j]) for k,v in a.items()});rec.append(r)
 csvout(O/'FIT_CLUSTER_BASE_PLANNING_ERRORS.csv',rows);csvout(O/'FIT_PAIR_BASE_PLANNING_ERRORS.csv',rec)
 js(O/'FIT_CLUSTER_BASE_PLANNING_ERRORS.json',dict(qualified_pair_population=1654,selection='Base S0 PDMS<70 and successful optimized post S0>90; outcome-conditioned subset, no full-population generalization',sources=[dict(path=str(p),sha256=sha(p)) for p in [fp,mp,tp]],rows=rows))
 for r in rows:
  if r['noise_aggregation']=='S0' and r['condition'] in ['all_fit_pairs','GT_left_turn']:print(json.dumps(r))

if __name__=='__main__':main()
