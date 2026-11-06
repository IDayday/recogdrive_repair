import recogdrive_paths as paths
from pathlib import Path
import csv,json
import numpy as np
from analyze_trajectories import csvout,js,sha
O=paths.analysis_root() / 'trajectories'

def main():
 fp=O.parent/'representation/FIT_PAIR_CLUSTERS.csv'
 with fp.open() as f:fit=list(csv.DictReader(f))
 mp=O.parent/'scene_metadata/train_metadata.npz'
 with np.load(mp,allow_pickle=False) as z:meta={k:z[k] for k in z.files}
 tp=O/'TRAIN_SELECTED_TRAJECTORIES.npz'
 with np.load(tp,allow_pickle=False) as z:tokens=z['tokens'];base=z['trajectories'][:,0,0].astype(np.float64)
 lookup={str(t):i for i,t in enumerate(tokens)};ix=np.asarray([lookup[r['token']] for r in fit]);cl=np.asarray([int(r['fit_cluster']) for r in fit]);b=base[ix];gt=meta['gt_trajectories'][ix].astype(np.float64)
 d=b[...,:2]-gt[...,:2];tan=np.stack([np.cos(gt[...,2]),np.sin(gt[...,2])],-1)
 err=(b[...,2]-gt[...,2]+np.pi)%(2*np.pi)-np.pi
 val={'GT_endpoint_alongtrack_error_m':(d*tan).sum(-1)[:,-1],'GT_end_y_error_m':d[:,-1,1],'GT_final_heading_error_deg':np.rad2deg(err[:,-1])}
 logs,inv=np.unique(meta['log_id'][ix],return_inverse=True);L=len(logs);rng=np.random.default_rng(20261105);draws=5000;w=rng.multinomial(L,np.full(L,1/L),size=draws)
 rows=[];difference=[]
 for label,cond in [('all_fit_pairs',np.ones(len(ix),bool)),('GT_left_turn_gt20deg',meta['gt_turn_deg'][ix]>20)]:
  boots={}
  for c in [1,2,3,5]:
   m=(cl==c)&cond;cnt=np.bincount(inv,weights=m,minlength=L);den=w@cnt;ok=den>0
   for k,v in val.items():
    sm=np.bincount(inv,weights=m*v,minlength=L);rep=np.full(draws,np.nan);rep[ok]=(w[ok]@sm)/den[ok];boots[(c,k)]=rep;lo,hi=np.nanquantile(rep,[.025,.975])
    rows.append(dict(fit_cluster=c,condition=label,noise='S0',metric=k,n=int(m.sum()),support_logs=int((cnt>0).sum()),mean=float(v[m].mean()),ci95_log_boot_low=float(lo),ci95_log_boot_high=float(hi)))
  for k in val:
   dff=boots[(5,k)]-boots[(3,k)];lo,hi=np.nanquantile(dff,[.025,.975]);m3=(cl==3)&cond;m5=(cl==5)&cond
   difference.append(dict(contrast='cluster5_minus_cluster3',condition=label,noise='S0',metric=k,n3=int(m3.sum()),n5=int(m5.sum()),mean=float(val[k][m5].mean()-val[k][m3].mean()),ci95_log_boot_low=float(lo),ci95_log_boot_high=float(hi)))
 csvout(O/'FIT_CLUSTER_S0_LOG_BOOTSTRAP.csv',rows);csvout(O/'FIT_CLUSTER3_VS5_S0_CONTRAST.csv',difference)
 js(O/'FIT_CLUSTER_S0_LOG_BOOTSTRAP.json',dict(draws=draws,seed=20261105,log_count=L,unit='physical log, original selected paired population only',rows=rows,contrasts=difference,sources=[dict(path=str(p),sha256=sha(p)) for p in [fp,mp,tp]]))
 for r in difference:print(json.dumps(r))
if __name__=='__main__':main()
