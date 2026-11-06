import recogdrive_paths as paths
"""Standalone publication figures from finite CPU summaries; no source mutation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=paths.analysis_root() / 'outcomes'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
s=json.loads((O/'SUMMARY.json').read_text())
branch=list(csv.DictReader((O/'BRANCH_PROFILES.csv').open()))
front=list(csv.DictReader((O/'TRAIN_PICKED_SUBSET_FRONTIER.csv').open()))
shap=list(csv.DictReader((O/'PDMS_ARITHMETIC_SHAPLEY.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(12,8.3))
ax=axs[0,0];x=np.arange(9)
for off,pop,col in [(-.18,'train','#3267a8'),(.18,'test','#d37d31')]:
 v=[float(r['scene_pct']) for r in branch if r['population']==pop]
 ax.bar(x+off,v,.36,label=pop.title(),color=col)
ax.set_xticks(x,['Base']+[f'E{i+1}' for i in range(8)]);ax.set_ylabel('Selected scenes (%)')
ax.set_title('(a) Similar branch-use profiles, Train vs Test')
ax.legend(frameon=False)
ax=axs[0,1]
cols=['#375d8a','#b67b35','#56a1a3','#70a15a','#b09bc7'];bottom=np.zeros(2)
for m,col in zip(['NC','DAC','TTC','Progress','Comfort'],cols):
 v=np.array([float(next(r['fraction_of_PDMS_gain'] for r in shap if r['population']==p and r['choice']=='ALL' and r['metric']==m))*100 for p in ['train','test']])
 ax.bar([0,1],v,.55,bottom=bottom,label=m,color=col);bottom+=v
ax.set_xticks([0,1],['Train','Test']);ax.set_ylabel('Arithmetic share of PDMS gain (%)')
ax.set_title('(b) Score gain: progress vs safety metrics')
ax.legend(frameon=False,ncol=2,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.1))
ax=axs[1,0]
for pop,key,col in [('Train','Train_gain_retained_pct','#3267a8'),('Test','Test_gain_retained_pct_same_subset','#d37d31')]:
 ax.plot([int(r['repair_choices_k']) for r in front],[float(r[key]) for r in front],'-o',color=col,label=pop)
ax.set_xlabel('Repair branches retained (Base always included)');ax.set_ylabel('Full-pool oracle gain retained (%)')
ax.set_xticks(range(9));ax.set_ylim(0,105);ax.set_title('(c) Complementarity; subset chosen on Train')
ax.legend(frameon=False);ax.grid(alpha=.2)
ax=axs[1,1]
for pop,col in [('train','#3267a8'),('test','#d37d31')]:
 with np.load(O/f'{pop.upper()}_OUTCOME_FEATURES.npz') as z:arr=z['nearopt1pt_count']
 freq=np.bincount(arr,minlength=10)[1:]/len(arr)*100
 ax.plot(range(1,10),np.cumsum(freq),'-o',label=pop.title(),color=col)
ax.set_xticks(range(1,10));ax.set_ylim(0,102)
ax.set_xlabel('Candidates within 1 point of the scene oracle');ax.set_ylabel('Cumulative fraction of scenes (%)')
ax.set_title('(d) Several near-optimal candidates per scene')
ax.legend(frameon=False);ax.grid(alpha=.2)
fig.suptitle('Local D / cluster-first / K8 / D48 / enhanced teacher repair',fontsize=14,y=.99)
fig.text(.5,.012,'Complete Train 85,813 / Test 12,146; five fixed noises per scene. Oracle diagnostics; not deployed routing.',ha='center',fontsize=9,color='#444')
fig.tight_layout(rect=(0,.035,1,.96),h_pad=3)
for ext in ['png','pdf','svg']:fig.savefig(O/f'OUTCOME_PATTERNS.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
