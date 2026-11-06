import recogdrive_paths as paths
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=paths.analysis_root() / 'trajectories'

def load(pop):
 with (O/f'{pop.upper()}_SCENE_GEOMETRY.csv').open() as f:r=list(csv.DictReader(f))
 keys=[k for k in r[0] if k not in ['population','token','method']]
 return {k:np.asarray([float(v[k]) for v in r]) for k in keys}

def write_csv(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 data={p:load(p) for p in ['train','test']};rows=[]
 for pop,d in data.items():
  choice=d['choice'].astype(int)
  # Buckets describe Base model plans, not annotated scene maneuvers.
  cat=np.select([d['Base_path_m']<2,d['Base_final_heading_deg']>20,d['Base_final_heading_deg']< -20,d['Base_abs_final_heading_deg']<10],['stall_path_lt2m','left_heading_gt20deg','right_heading_ltminus20deg','straight_heading_lt10deg'],default='moderate_turn')
  for label in np.unique(cat):
   m=cat==label
   for c in range(9):
    g=m&(choice==c)
    row=dict(population=pop,Base_plan_bucket=label,choice=c,slot=c-1,bucket_n=int(m.sum()),selected_n=int(g.sum()),selected_share_within_bucket_pct=float(g.sum()/m.sum()*100),gain_points_mean=float(d['gain_points'][g].mean()) if g.any() else None,delta_endpoint_x=float(d['endpoint_dx'][g].mean()) if g.any() else None,delta_path_m=float(d['delta_path_m'][g].mean()) if g.any() else None)
    rows.append(row)
 write_csv('BASE_PLAN_BUCKET_SELECTIONS.csv',rows)
 compare=[]
 for c in range(1,9):
  row=dict(choice=c,slot=c-1)
  for pop,d in data.items():
   m=d['choice']==c
   row[pop+'_n']=int(m.sum())
   for k in ['Base_path_m','Base_mean_speed_mps','Base_final_heading_deg','Base_PDMS','gain_points','ADE_m','FDE_m','endpoint_dx','endpoint_dy','delta_path_m','delta_abs_final_heading_deg','delta_mean_abs_curvature']:
    row[pop+'_'+k]=float(d[k][m].mean())
   for k,m2 in [('forward_extension',d['endpoint_dx']>.5),('forward_shortening',d['endpoint_dx']<-.5),('negligible_forward_change',np.abs(d['endpoint_dx'])<=.5),('left_shift',d['endpoint_dy']>.5),('right_shift',d['endpoint_dy']<-.5)]:row[pop+'_'+k+'_pct']=float(m2[m].mean()*100)
  compare.append(row)
 write_csv('TRAIN_TEST_GEOMETRY_TRANSFER.csv',compare)
 fig,axs=plt.subplots(2,3,figsize=(14,8))
 columns=[('endpoint_dx','End forward displacement change [m]'),('endpoint_dy','End lateral displacement change [m]'),('delta_path_m','Path length change [m]'),('ADE_m','Mean trajectory distance to Base [m]'),('Base_mean_speed_mps','Base mean planned speed [m/s]'),('gain_points','Oracle-selected PDMS gain [points]')]
 x=np.arange(8)
 for ax,(k,title) in zip(axs.ravel(),columns):
  for off,p,col in [(-.18,'train','#465b74'),(.18,'test','#198b78')]:ax.bar(x+off,[r[p+'_'+k] for r in compare],width=.36,label=p,color=col)
  ax.axhline(0,color='black',lw=.6);ax.set_title(title,fontsize=10);ax.set_xticks(x,[str(i) for i in range(8)]);ax.set_xlabel('frozen expert slot');ax.grid(axis='y',alpha=.2)
 axs[0,0].legend();fig.suptitle('Selected-branch profiles: full Train vs full Test (descriptive oracle groups)');fig.tight_layout();fig.savefig(O/'TRAIN_TEST_BEHAVIOR_PROFILES.png',dpi=180);fig.savefig(O/'TRAIN_TEST_BEHAVIOR_PROFILES.pdf');plt.close(fig)
 summary={}
 for pop,d in data.items():
  m=d['choice']>0
  summary[pop]=dict(n=len(m),repaired_n=int(m.sum()),endpoint_dx_positive_gt0p5_pct=float((d['endpoint_dx'][m]>.5).mean()*100),endpoint_dx_negative_ltminus0p5_pct=float((d['endpoint_dx'][m]<-.5).mean()*100),path_length_increase_gt0p5_pct=float((d['delta_path_m'][m]>.5).mean()*100),heading_abs_decrease_gt2deg_pct=float((d['delta_abs_final_heading_deg'][m]<-2).mean()*100),forward_vs_gain_pearson=float(np.corrcoef(d['endpoint_dx'][m],d['gain_points'][m])[0,1]),ADE_vs_gain_pearson=float(np.corrcoef(d['ADE_m'][m],d['gain_points'][m])[0,1]))
 (O/'TRANSFER_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary))

if __name__=='__main__':main()
