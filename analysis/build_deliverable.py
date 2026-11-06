"""Build figures and a traceable paper-evidence report from completed CPU analyses."""
from __future__ import annotations
import recogdrive_paths as paths
import csv
import gzip
import hashlib
import html
import json
import pickle
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

ROOT=paths.analysis_root()
N=paths.native_root()
CODE=paths.code_root()


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def jread(path):return json.loads(path.read_text())


def jwrite(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(c) for c in row)+' |\n' for row in rows)


def savefig(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/f'{name}.{suffix}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def make_figures():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    (ROOT/'figures').mkdir(exist_ok=True)
    rows=read_csv(ROOT/'representation/FIT_CLUSTER_SCENE_PROFILES.csv')
    x=np.arange(8)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8),gridspec_kw={'width_ratios':[1.3,1,1]})
    bottom=np.zeros(8)
    for name,color in [('straight','#5998c5'),('left','#51a786'),('right','#dc9b57'),('stationary','#bbb')]:
        v=np.asarray([float(r[f'gt_motion__{name}'])*100 for r in rows])
        ax[0].bar(x,v,bottom=bottom,label=name,color=color);bottom+=v
    ax[0].set(xlabel='Original fit cluster C0-C7 (1654 successful pairs)',ylabel='GT motion composition (%)',xticks=x,ylim=(0,100))
    ax[0].legend(fontsize=8,ncol=2,loc='upper center')
    v=[float(r['ego_speed_mps_mean']) for r in rows]
    ax[1].bar(x,v,color='#487d9e');ax[1].set(xlabel='Fit cluster',ylabel='Current ego speed (m/s)',xticks=x)
    ax[2].bar(x-.18,[float(r['gt_start_fraction'])*100 for r in rows],.36,label='GT start',color='#6aaf87')
    ax[2].bar(x+.18,[float(r['gt_brake_to_stop_fraction'])*100 for r in rows],.36,label='GT brake to stop',color='#c98260')
    ax[2].set(xlabel='Fit cluster',ylabel='Diagnostic context rate (%)',xticks=x);ax[2].legend(fontsize=8)
    fig.suptitle('Repair-direction clusters contain motion context, but are not pure semantic classes')
    fig.tight_layout();savefig(fig,'CLUSTER_SCENE_SEMANTICS')
    enrich=read_csv(ROOT/'scene_metadata/CHOICE_SEMANTIC_ENRICHMENTS.csv')
    names=['low_speed_lt2','gt_start','gt_brake_to_stop','front_stopped_vehicle20','command_right','dense_vehicles20_ge4']
    labels=['Speed <2 m/s','GT start','GT brake to stop','Front stopped vehicle','Right command','>=4 vehicles / 20 m']
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.3),sharey=True)
    for ax,pop in zip(axes,['train','test']):
        lookup={(int(r['choice']),r['variable']):r for r in enrich if r['population']==pop}
        a=np.asarray([[100*float(lookup[(i,k)]['difference']) for k in names] for i in range(9)])
        im=ax.imshow(a,cmap='RdBu_r',vmin=-50,vmax=50,aspect='auto')
        ax.set_xticks(np.arange(len(names)));ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=8)
        ax.set_yticks(np.arange(9));ax.set_yticklabels(['Base']+[f'E{i}' for i in range(1,9)])
        ax.set_title(pop.capitalize()+' (all scenes, selected groups)')
        for i in range(9):
            for k,name in enumerate(names):
                r=lookup[(i,name)];flag=float(r['log_bootstrap_diff_low'])*float(r['log_bootstrap_diff_high'])>0
                ax.text(k,i,f'{a[i,k]:+.0f}'+('*' if flag else ''),ha='center',va='center',fontsize=8,color='white' if abs(a[i,k])>28 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.022,pad=.025,label='Group rate minus other scenes (percentage points)')
    fig.suptitle('Cross-split phenotypes of selected branches; * pointwise log-bootstrap 95% CI excludes 0')
    fig.subplots_adjust(bottom=.29,top=.86,wspace=.12,right=.9);savefig(fig,'TRAIN_TEST_BRANCH_PHENOTYPES')
    rows=read_csv(ROOT/'trajectories/FIT_CLUSTER_S0_LOG_BOOTSTRAP.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.7))
    for ax,(key,label) in zip(axes,[('GT_endpoint_alongtrack_error_m','Endpoint along-GT error (m)'),
                                   ('GT_end_y_error_m','Endpoint ego-y error (m)'),
                                   ('GT_final_heading_error_deg','Final heading error (degrees)')]):
        pair=[next(r for r in rows if r['fit_cluster']==str(c) and r['condition']=='GT_left_turn_gt20deg' and r['metric']==key) for c in [3,5]]
        means=np.asarray([float(r['mean']) for r in pair])
        lows=np.asarray([float(r['ci95_log_boot_low']) for r in pair]);highs=np.asarray([float(r['ci95_log_boot_high']) for r in pair])
        ax.bar([0,1],means,color=['#548ca5','#cd8161'],width=.55)
        ax.errorbar([0,1],means,yerr=[means-lows,highs-means],fmt='none',ecolor='#222',capsize=5)
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks([0,1]);ax.set_xticklabels(['C3: n=81 / 35 logs','C5: n=108 / 49 logs'],fontsize=8)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Same GT left-turn context, opposite Base planning errors (S0; 95% log-bootstrap CI)')
    fig.text(.5,.005,'Descriptive subsets: speed, city and GT path length differ; this is not a matched causal comparison.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.96]);savefig(fig,'SAME_TURN_DIFFERENT_ERRORS')


def qualitative():
    out=ROOT/'cases';out.mkdir(exist_ok=True)
    allcases=[]
    for pop in ['train','test']:
        with np.load(ROOT/'scene_metadata'/f'{pop}_metadata.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
        with np.load(N/'frozen_teacher/selections'/f'{pop}.npz',allow_pickle=False) as z:tokens,q,choice=z['tokens'],z['q_points'],z['choice']
        if pop=='train':
            with np.load(ROOT/'trajectories/TRAIN_SELECTED_TRAJECTORIES.npz',allow_pickle=False) as z:
                assert np.array_equal(tokens,z['tokens']);traj=z['trajectories']
            base,chosen=traj[:,0],traj[:,1]
        else:
            with np.load(N/'engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz',allow_pickle=False) as z:
                assert np.array_equal(tokens,z['tokens']);traj=z['trajectories']
            base=traj[:,0];chosen=traj[np.arange(len(choice)),choice]
        with gzip.open(CODE/'manifests'/f'{pop}.csv.gz','rt') as f:source={r['token']:r for r in csv.DictReader(f)}
        gain=q.max(1)-q[:,0]
        definitions=[(2,'Front stopped vehicle',m['front_stopped_vehicle20']),
                     (4,'Right navigation command',m['command']=='right'),
                     (7,'GT start',m['gt_start']),
                     (8,'Cruising, no front stopped vehicle',(m['ego_speed_mps']>=5)&(~m['front_stopped_vehicle20']))]
        fig,axes=plt.subplots(4,2,figsize=(12,13),gridspec_kw={'width_ratios':[1.4,1]})
        for line,(j,description,feature) in enumerate(definitions):
            keep=(choice==j)&feature&(gain>1)&(m['future_dt_max_error_s']<=.1)
            pool=np.flatnonzero(keep)
            assert len(pool)>0
            feat=np.column_stack([gain[pool],m['ego_speed_mps'][pool],m['gt_turn_deg'][pool]])
            scales=np.maximum(np.subtract(*np.percentile(feat,[75,25],axis=0)),[1,1,5])
            dist=np.sum(np.abs((feat-np.median(feat,axis=0))/scales),axis=1)
            i=int(pool[np.argmin(dist)]);token=str(tokens[i]);row=source[token]
            rawbase=paths.logs_root(pop)
            rawpath=rawbase/row['raw_rel'];d=pickle.loads(rawpath.read_bytes())[int(row['frame_index'])]
            imagepath=paths.sensor_root(pop)/row['sensor_rel']/d['cams']['CAM_F0']['data_path']
            with Image.open(imagepath) as img:
                img=img.convert('RGB');img.thumbnail((900,520));img.save(out/f'{pop}_{token}.jpg',quality=88)
                axes[line,0].imshow(img)
            axes[line,0].set_title(f'{pop.upper()} E{j}: {description}\n{token} | speed={m["ego_speed_mps"][i]:.2f} m/s',fontsize=10)
            axes[line,0].axis('off')
            ax=axes[line,1]
            boxes=np.asarray(d['anns']['gt_boxes']);names=np.asarray(d['anns']['gt_names']).astype(str)
            for box,name in zip(boxes,names):
                if np.linalg.norm(box[:2])>45:continue
                length,width=box[3:5];corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*np.array([length,width])/2
                co,si=np.cos(box[6]),np.sin(box[6]);xy=corners@np.array([[co,si],[-si,co]])+box[:2]
                ax.add_patch(Polygon(np.column_stack([-xy[:,1],xy[:,0]]),facecolor='#d19b62' if name=='pedestrian' else '#bbb',edgecolor='#888',alpha=.35,lw=.5))
            gt=m['gt_trajectories'][i]
            for arr,color,label in [(base[i,0],'#444','Base S0'),(chosen[i,0],'#d4533d','Teacher S0'),(gt,'#318967','Human GT')]:
                xy=np.vstack([np.zeros((1,3)),arr]);ax.plot(-xy[:,1],xy[:,0],'-o',ms=2,lw=1.6,label=label,color=color)
            pts=np.concatenate([base[i,0,:,:2],chosen[i,0,:,:2],gt[:,:2],np.zeros((1,2))])
            low=pts.min(0);high=pts.max(0)
            ax.set_xlim(-high[1]-4,-low[1]+4);ax.set_ylim(min(-2,low[0]-2),high[0]+4)
            ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
            ax.set(xlabel='Right = -ego y (m)',ylabel='Forward ego x (m)')
            ax.set_title(f'5-noise PDMS: {q[i,0]:.2f} -> {q[i,j]:.2f}\nDelta {gain[i]:+.2f}; S0 waypoints shown',fontsize=10)
            ax.legend(fontsize=7,loc='upper left')
            case=dict(population=pop,token=token,choice=j,slot=j-1,description=description,
                      candidate_group_size=int(len(pool)),selection_rule='Within chosen branch and specified phenotype, gain >1 and timestamp error <=100ms; minimize L1 distance to median gain/speed/GT turn after IQR scaling; no manual selection',
                      Base_PDMS=float(q[i,0]),teacher_PDMS=float(q[i,j]),all_candidate_PDMS=q[i].tolist(),
                      ego_speed_mps=float(m['ego_speed_mps'][i]),GT_turn_deg=float(m['gt_turn_deg'][i]),
                      Base_trajectory_S0=base[i,0].tolist(),teacher_trajectory_S0=chosen[i,0].tolist(),GT=gt.tolist(),
                      raw_log=dict(path=str(rawpath),sha256=digest(rawpath)),
                      original_front_image=dict(path=str(imagepath),sha256=digest(imagepath)),
                      thumbnail=f'{pop}_{token}.jpg')
            allcases.append(case)
        fig.suptitle('Diagnostic median-context examples; not a random sample or proof of universal specialist behavior',fontsize=11,y=.995)
        fig.tight_layout();fig.savefig(out/f'{pop.upper()}_QUALITATIVE_CASES.png',bbox_inches='tight',dpi=165)
        fig.savefig(out/f'{pop.upper()}_QUALITATIVE_CASES.pdf',bbox_inches='tight');plt.close(fig)
    jwrite(out/'CASES.json',allcases)


def report():
    outcomes=jread(ROOT/'outcomes/SUMMARY.json')
    fit=read_csv(ROOT/'representation/FIT_CLUSTER_SCENE_PROFILES.csv')
    branches=read_csv(ROOT/'outcomes/BRANCH_PROFILES.csv')
    profiles={p:read_csv(ROOT/'scene_metadata'/f'{p}_choice_profiles.csv') for p in ['train','test']}
    fitrows=[]
    for r in fit:
        fitrows.append([f'C{r["cluster"]}',r['n'],f'E{r["matched_frozen_choice"]}',f'{float(r["ego_speed_mps_mean"]):.2f}',
                        f'{100*float(r["gt_motion__straight"]):.1f}%',f'{100*float(r["gt_motion__left"]):.1f}%',
                        f'{100*float(r["gt_motion__right"]):.1f}%',f'{100*float(r["gt_start_fraction"]):.1f}%',
                        f'{100*float(r["gt_brake_to_stop_fraction"]):.1f}%'])
    branchtable=[]
    for j in range(1,9):
        a=next(r for r in branches if r['population']=='train' and int(r['choice'])==j)
        b=next(r for r in branches if r['population']=='test' and int(r['choice'])==j)
        branchtable.append([f'E{j}',f'{float(a["scene_pct"]):.2f}% / {float(b["scene_pct"]):.2f}%',
                            f'{float(a["selected_gain"]):.2f} / {float(b["selected_gain"]):.2f}',
                            f'{float(b["all_scenes_fixed_branch_delta"]):+.2f}',
                            f'{float(b["delete_this_branch_loss_PDMS"]):.3f}'])
    root=ROOT
    text='''# ReCogDrive 表征修复规律挖掘：场景、规划行为与表征机制

本报告基于已完成的本机 D／先聚类／K8／D48／增强教师实验，重新联接原始 NAVSIM 场景与已有模型输出，完成纯 CPU 分析。没有重新拟合教师、重新评分、启动 GPU 任务或恢复已停止的蒸馏。分析覆盖完整 Train **85,813 场景／130 会话**、完整 Test **12,146 场景／44 会话**，并单独研究用于拟合的 **1,654 个成功 GT 配对／121 会话**。

## 1. 直接回答：有规律，但应分清三种标签

**有运动情境与规划偏差的共性；目前不足以把八簇命名为八种缺失能力。**

1. **拟合簇 C0–C7**：对成功接口与失败 Base 接口的有效 token 池化共享 384 维视图差分单位方向聚类，只存在于 1,654 个定向成功配对中。C编号指本机 `ALLTRAIN/SPACE_AND_CLUSTER.pt` 的重拟合 `labels/new_cluster`，不是内层选择配置中的 cluster 字段。先在 384 维聚类，再使用共享 D48 子空间，不是对所有场景图像做语义聚类。
2. **最终修复分支 E1–E8**：分别对应实现中的 slot0–slot7；0 为 Base。每个完整 Train/Test 场景实际执行 Base＋8 个分支，以 S0–S4 五噪声平均真实 PDMS 选择一个固定分支。最终 choice 并不等于原拟合簇。
3. **修复算子名称**：五个不同分支都叫 `centroid_D0`，但方向不同，甚至彼此相反。不能把同算子名等同于同修复方案或同能力。

下面的 E 编号始终从 1 开始，C 编号从 0 开始；避免把 slot7 与 E7 混淆，slot7 实际是 E8。

## 2. 原聚类确实包含哪些场景共性

只在 1,654 个源配对中统计，所有 GT 属性只作离线诊断。GT 运动按未来八帧轨迹判定：路径小于 2m 为静止，其余末端 heading 大于 +15°／小于 −15° 为左转／右转，其余为直行。起步定义当前速度 <0.5m/s 且未来路径 >5m；刹停定义当前速度 >2m/s 且第八帧速度 <1m/s。

'''+table(['拟合簇','配对数','对应专家','当前均速m/s','GT直行','GT左转','GT右转','GT起步','GT刹停'],fitrows)+'''
据此可描述：C0/C7 偏高速直行；C1 偏低速与起步；C2 明显富集刹停；C3/C5 都偏左转；C6 同时包含大量左右转。**这不是一簇一个路况的纯类别。**

效应量：拟合簇与导航指令 Cramér’s V=**0.430**，与 GT 运动 V=**0.416**，与速度分箱 V=**0.388**；速度组间 η²=**0.319**，GT 路径长度 η²=**0.289**。导航指令和 GT 运动分别只解释约 9.5% 和 12.8% 的簇标签熵，说明共性真实存在，却远不足以还原所有簇。

![原聚类场景画像](figures/CLUSTER_SCENE_SEMANTICS.png)

### 同为左转，为什么还会分到不同簇

同样是 GT 左转 >20°的源配对，C3/C5 分别有 81／108 例、来自35／49会话，平均 GT 转角约 **44.65°／43.40°**，但 Base S0 规划转角约 **38.06°／46.76°**；末端 heading 误差约 **−6.59°／+3.36°**，沿 GT 方向的终点误差约 **−0.710m／+2.054m**。C5−C3 的沿程误差差值 **2.765m**，会话 bootstrap 95% CI **[1.813,3.742]m**；heading差 **9.95°**，CI **[4.96,13.57]°**。

这提示方向聚类不仅反映“在左转”，也包含同类运动情境下的不同规划偏差。不能只靠“左转”一个标签决定修复方向。此处是低分成功配对内的描述，不是模型缺陷的因果分型；两簇当前速度 **5.77／4.71m/s**、GT路径 **21.41／18.09m**、Singapore比例 **50.62%／26.85%** 不同，存在混杂。置信区间和全部误差见 `trajectories/FIT_CLUSTER3_VS5_S0_CONTRAST.csv` 与 `FIT_CLUSTER_S0_LOG_BOOTSTRAP.csv`，不能将五噪声当独立场景。

![相同左转情境中的不同规划偏差](figures/SAME_TURN_DIFFERENT_ERRORS.png)

### 不能把聚类标签直接用作专家选择

源配对所属簇对应的专家，与最终 Oracle 精确选择仅 **408/1654=24.67%** 一致，即使计入并列最优也只有 **36.09%**。在相同 1,654 场景上：Base **35.64**，直接使用所属簇专家 **63.83**，全候选 Oracle **91.11**。这个差距不是新的部署实验；它是已有全部候选评分的诊断重组，而且簇标签本身需要 GT 优化差分，不是部署可得输入。

## 3. 同一修复分支在 Train/Test 上有什么共同规律

以下为**选择后人群画像**：描述哪些场景最终由该分支胜出，不代表该分支可无条件处理该类场景。

| 分支 | 跨 Train/Test 重复的画像 | Train／Test 实测 | 可支持的解释 |
| --- | --- | --- | --- |
| E2（slot1，centroid） | 前方停驶车辆、GT刹停 | 前方停驶车 58.28%／38.16%；GT刹停 55.13%／40.79% | 与减速停车情境有关，尚非统一制动能力模块 |
| E4（slot3，centroid） | 右转导航与右转规划 | 右转命令 28.50%／40.50%；较其余场景分别高 18.37／29.59 个百分点 | 选择后的转向相关表型 |
| E7（slot6，centroid） | 低速与GT起步 | GT起步 19.85%／21.00%，其余场景仅 6.54%／5.49% | 与低速起步情境有关，不能直接称起步迟滞修复 |
| E8（slot7，mean） | 较快行驶、周边较稀疏 | 当前均速 7.07／7.14m/s；选后规划路径增加 1.68／1.71m | 改变沿程与评分偏好，不等于更贴近人类轨迹 |

E2 的前方停驶车富集相对其余场景约 **4.57／3.26 倍**；E4 右转命令约 **2.81／3.71 倍**；E7 起步约 **3.03／3.83 倍**。这些差异的逐会话 bootstrap 95% 点区间在两个 split 均不跨零，详见 `scene_metadata/CHOICE_SEMANTIC_ENRICHMENTS.csv`。它们是探索性、非多重比较校正后的发现，不能据此宣布多项独立显著能力。

![最终分支的跨域画像](figures/TRAIN_TEST_BRANCH_PHENOTYPES.png)

八个修复分支使用率在 Train/Test 间 Pearson=**0.9986**，包含 Base 的选择分布 JS=**0.00672 bits**；各分支条件收益相关 **0.9687**。同分支的 NC/DAC/TTC/Progress 改变量也高度同向。但只有八个相关分支、一个冻结模型，这些相关不是独立重复实验。

### 关键反证：场景画像不能直接变成路由规则

尽管 E4 选中场景富集右转，**把 E4 用于全部右转命令场景**会使 PDMS 相对 Base 下降 **Train −20.90／Test −18.96**。E2 用于全部“前方停驶车辆”场景也会下降 **−4.78／−3.52**。这不是文字解释，而是对这些外生场景分组读取已有固定候选评分的结果，没有再次用 winner 过滤。

进一步按城市×当前速度×Base分数分箱控制，E2 在前方停驶车群体相对其他修复分支的比较优势仍更高，但绝对效果依旧不能替代 Base。GT 起步等调整有共同支持不足，已报告覆盖率，不强行推断。**“该类场景常选它”与“看到该类场景就该选它”完全不同。**

'''+table(['分支','选择率Train／Test','选中人群平均增益Train／Test','全Test固定应用相对Base','删除该分支Test损失'],branchtable)+'''
表中的正条件增益来自事后最优分支选择，不能当作独立能力证据；最后两列分别衡量普遍应用的风险及候选的边际互补价值。

## 4. 模型规划究竟发生了什么变化

在实际选中修复的 Train 46,573／Test 7,574 场景中，Base→教师轨迹 ADE 约 **0.776／0.795m**，FDE **1.665／1.763m**；末点前向平均增加 **1.372／1.192m**。按会话 bootstrap，前向变化 95% CI 为 **[1.341,1.402]／[1.106,1.277]m**。

但沿前向移动更多不等于 PDMS 收益更大：修复群体末点 Δx 与 gain 的 Pearson 约 **−0.284／−0.472**。它混合了大量小收益的沿程改变，以及少数幅度更复杂的安全纠错，不能简化成“整体多往前开一点”。

### 提分不等于恢复人类GT，也不能直接证明原模型过度保守

完整 Train 的人类 GT-ADE 从 **0.641→0.869m**，完整 Test 从 **0.829→0.989m**，同时 PDMS 从 **95.07→97.53／90.68→96.55**。

在实际修复人群中，GT-ADE 为 **0.671→1.091m／0.847→1.103m**；**87.33%／72.08% 的正 PDMS 场景反而 GT-ADE 变差**。E8 的 Base 沿 GT 末端方向平均本已超前 **0.751／0.651m**，教师后变为 **2.455／2.374m**。因此，“E8 修复了相对人类轨迹的欠进展／过度保守”被这组证据反驳。

更准确的解释是：**同一连续表征干预可改变下游规划的行为偏置，获得 PDMS 更偏好的轨迹；它不必恢复人类示范轨迹或某个可观测的正确语义表征。** 这仍是机制解释，不是证明 PDMS 奖励黑客，也不等于更远离 GT 就更好。

GT 是原始未来八帧，与原 NAVSIM 序列定义一致；少量时间戳缺帧导致其并非全都严格 4秒。Train 560例（0.65%）／Test 36例（0.30%）的最大时序误差超过100ms，已单列并做剔除敏感性检查，核心方向不因这些少量样本改变。几何分析使用原生预测 waypoints，不冒充 PDM 仿真后的闭环 ego 轨迹。

## 5. 哪些错误贡献了主要收益

教师完整 Train／Test 增益分别 **+2.4577／+5.8752 PDMS**，是 Oracle 上限，不是学生或可部署路由结果。

- Test 只有 **962场景、7.92%** 在任一噪声中出现 Base NC/DAC/TTC 失败，却贡献总增益的 **76.72%**。Train 对应 **2,021场景、2.36%**，贡献 **47.84%**。安全失败项可重叠，不能把三类人数直接相加。
- 在 NC/DAC/TTC 五噪声全部通过、但进度未满的场景，Train／Test 条件收益分别 **+2.3564／+2.3393**，两侧相近。
- 固定评分公式的精确算术 Shapley 分解表明，Progress 占 Train 增益 **58.46%**、Test **34.36%**；Test 的 NC/DAC/TTC 合计 **65.64%**。这是分数公式的代数贡献，不是模型能力的因果贡献。
- 相同 Base PDMS 五档的组成分解中，两侧增益差的 **91.87%** 对应难度人口组成差异。它说明 Test 提分更大很大程度上伴随更大的可修复空间；不能写成已证明修复对未见数据的泛化优于训练数据。
- **DDC 反而下降 Train 1.2503／Test 1.0884 个百分点**，会话 bootstrap CI 均为负。当前 DDC 权重为0，因此 PDMS 提分没有排除这个代价。不能声称所有驾驶能力都同步提升。

原始安全恢复计数、全部指标、区间和组成分解见 `outcomes/`，并附精确评分重建验证。

## 6. 从表征层面能支持什么规律

1. D48 只占 384 维的 **12.5%**，捕获源配对归一化修复差分能量的 **62.62%**、原始差分能量 **66.25%**。这是低维能量集中，不是“错误只有48维”或“48维已足够”的证明；没有新增匹配 PDMS 的 rank 消融。
2. 修复方向的 cosine silhouette 仅 **0.0764**；同标签在 Base pooled 表征、普通 state24 的 silhouette 为负。20次重启之间 ARI 均值 **0.549**，说明存在结构，但边界不是唯一自然类别。低 silhouette 与前述运动属性富集并不矛盾。
3. 最终八专家是 **1 contract＋5 centroid_D0＋1 local＋1 mean**，其中六个封顶前为固定向量。选中修复中 **Train 75.64%／Test 75.22%** 复用这六个固定向量；这部分相同方向是构造结果，不作为跨域语义发现。
4. 全部实际选中非零 shift 的前八个主方向解释 **93.50%／93.16%** 的**未中心化SVD能量**，不是中心化PCA方差；其中大部分集中性来自专家结构和选择频率，不能当作独立发现了“八维驾驶智能”。
5. `centroid` 同名向量也可能相反，例如 E2 与 E4 余弦 **−0.792**。动态 contract 在两侧所选人群的平均方向余弦约 **0.869**，local 约 **0.973**。这说明条件人群平均方向的跨split稳定性不同，受人群构成影响；二者本身均为输入相关算子，不能仅用这两个余弦推断其输入依赖程度。

这里的 `actual_role_shifts` 是原生 cap 之后实际施加的 image/text 向量，两者逐值相同；`actual_RMS` 是 FP32 加法后**有效token**差值的 RMS，不能误称含 padding 的全token RMS。padding 保持原值。源优化缓存 `actual_fulltoken_RMS` 的历史字段名另作档案引用，不能与上述新口径混用。

## 7. 互补性、近并列与选择难度

从 Train 选择五个分支 **E2/E4/E6/E7/E8**，在 Test 仍保留完整候选 Oracle 收益的 **93.93%**。只保留 Train 最优的单修复 E8＋Base，则 Test 只保留 **29.03%** 收益。这支持“不同方向存在互补性”，没有证明部署能选对。

平均每场景有 **Train 4.365／Test 3.798** 个候选距最优不超过1分；最优对次优间隔>1只有 **25.01%／29.37%**。因此硬专家编号包含大量近并列任意性，未来分析选择器更宜看 regret／top-k／近优集合，而不只看精确分类准确率。

另一方面，固定选择在五次噪声均为并列最优的场景仍有 **91.06%／88.55%**，逐噪声重新选专家只再提高 **0.070／0.090** 分左右（精确值以 `outcomes/SUMMARY.json` 为准）。当前收益不是依靠每次噪声随意更换教师得到。

Base-first 获胜 Train45.73%／Test37.64%，但从候选池删掉 Base，Oracle 仅损失 **0.0332／0.0381 PDMS**。所以“约40%场景选Base”本身不足以论证必须增加一个修复开关分类器；很多是并列或可替代的成功轨迹。

## 8. 论文主张：证据级别与适合的表述

| 证据级别 | 可以写什么 | 不应升级成什么 |
| --- | --- | --- |
| 已测量 | 修复差分低维集中，方向簇与运动情境及部分规划偏差相关 | 八簇对应八种确定、互斥的驾驶能力 |
| 已测量 | 相同左转情境出现不同的 Base 转向／沿程误差 | 这些误差一定由感知缺失或注意力错误导致 |
| 已测量 | Train/Test 共享分支使用和场景画像；候选互补 | 一个简单语义分类器已经可实现教师上限 |
| 已测量 | 少量严重安全错误贡献大量 Test 收益；指标存在代价 | PDMS 提高代表所有安全、效率、遵规能力同步改善 |
| 解释性假设 | 修复可改变冻结下游规划的行为偏置，适用性判断很关键 | 已证明模型缺少具体物体识别、因果推理或交通规则知识 |

可用于论文的归纳：

> 在冻结 ReCogDrive 接口上，成功规划修复的表征差分具有低维能量集中，并包含与速度、转向和制动情境相关的结构。但这些方向簇并非自然场景类别，也不能直接决定最优干预。相同运动情境可出现不同规划误差，修复方向的适用性仍显著依赖上下文。训练与测试中重复出现的响应画像、候选间的互补性，以及 PDMS 与人类轨迹误差的分离，共同表明该接口可用于有条件的规划行为干预，而不足以将修复解释为恢复某种已确定缺失的语义能力。

若要进一步证明“缺什么能力”，需要独立于 winner 的人工场景标签、同情境的匹配干预、逐对象或逐信息源消融，以及新的未用于研究决策的评测集。本轮未自动启动这些额外实验。

## 9. 定性样例与可核查性

定性例按明确规则自动选择：每个分支预设一种情境，保留 gain>1、时间戳偏差≤100ms，在其中选 gain／速度／GT转角接近该组中位数的场景。不是随机总体样本，也没有手工挑选最漂亮结果。每例记录完整token、所有候选均分、原图和原日志 SHA；图中画 S0 轨迹，标题显示五噪声均分，两者明确区分。

![Test真实场景和轨迹](cases/TEST_QUALITATIVE_CASES.png)

![Train真实场景和轨迹](cases/TRAIN_QUALITATIVE_CASES.png)

另见 `trajectories/REPRESENTATIVE_TRAJECTORIES.png`，包含按几何与评分规则选择的正反案例。所有图的灰色框仅为原始 ego 坐标交通参与者标注，不是 HD 地图或新感知预测。

## 10. 来源、统计边界与复现

- 教师：`frozen_teacher/BUNDLE.json` SHA `ed0a1bef8b27d89e7bb4eb8f32c93134204d66c42184ddecaa1831c5a7188dfc`；RL Base SHA `23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`。只分析本地重建教师，不混入 PPU 轨迹。
- 原始完整评分 `requests/train|test/MATRIX.npz`，固定选择 `frozen_teacher/selections/train|test.npz`；全部源帧逐token与冻结清单核对。原始日志共 Train1008／Test136文件，逐文件哈希记录在 `scene_metadata/*_raw_sources.json`。
- 五噪声先在场景内配对平均；没有将 S0–S4 当五个训练种子。区间按原始130／44会话重采样，保留同会话关联；其不覆盖模型训练种子或采集地点的全部不确定性。
- 语义、规划、机制与收益的统计人口不同，表格均标明；方向簇只在1654定向成功配对上存在，不把它的分布冒充完整Train/Test。
- Test 在历史开发中已用于分析与选择，本报告是探索性跨split一致性描述，**不是新的盲测或确认性泛化检验**。
- 没有根据本轮分析重新选教师参数、重写标签、改变采样、改变评分或运行部署路由。被选人群的正收益受Oracle定义影响，另给所有外生分组的固定候选效果作对照。

所有脚本、表格、图与来源 SHA 见 `MANIFEST.json`；`NUMERIC_CHECKS.json` 记录主agent的独立一致性复核。`PAPER_CLAIMS.csv` 逐条列出建议论文主张、统计人口、证据文件和结论边界。原教师与历史资产没有修改。

主要复现命令（均为已有数据上的 CPU 分析）：

```bash
export OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 analyze_scene_metadata.py
python3 semantic_followup.py
python3 representation/analyze_mechanisms.py
# representation/ 追加分析与 trajectories/ 的脚本/来源详见各自报告
python3 outcomes/analyze_outcomes.py
python3 outcomes/supplement_outcomes.py
python3 build_deliverable.py
```

八个旧蒸馏任务仍保持用户要求的停止状态，本报告不代表重新完成或恢复这些训练。
'''
    # Use source exact value rather than rounded prose placeholder for the noise comparison.
    exact=outcomes['test']['single_noise_oracle_advantage_over_fixed']
    text=text.replace('**0.070／0.090** 分左右',f'**0.070／{exact:.3f}** 分左右')
    (ROOT/'REPAIR_PATTERN_REPORT.md').write_text(text)
    (ROOT/'README.md').write_text('''# ReCogDrive 修复规律分析证据包

入口：[完整分析报告](REPAIR_PATTERN_REPORT.md)。覆盖完整 Train85813／Test12146，以及1654个源拟合配对；纯CPU读现有资产，没有恢复训练。

- `representation/`：方向簇、低维能量、算子、源配对语义、簇与最终选择的关系。
- `scene_metadata/`：完整原始场景元数据、跨split画像、会话bootstrap、外生分组固定候选效果。
- `trajectories/`：完整配对轨迹几何、GT误差、源簇规划偏差及正反样例。
- `outcomes/`：分支收益、安全恢复、选择稳定性、子集消融、评分代数与难度组成。
- `figures/` 与 `cases/`：可导出的PDF/PNG/SVG图、带原始图像的Train/Test案例和来源SHA。
- `NUMERIC_CHECKS.json`、`MANIFEST.json`：独立一致性复核与产物哈希。

编号统一：Base=0，E1–E8=choice1–8=slot0–7；源拟合簇单独标C0–C7。
紧凑压缩包不含大NPZ与完整逐场景CSV；这些全量数据保留在本目录，聚合表和代码随包提供。
''')
    sections=[]
    for title,rel in [('原聚类的场景结构','figures/CLUSTER_SCENE_SEMANTICS.png'),
                      ('相同左转的不同规划偏差','figures/SAME_TURN_DIFFERENT_ERRORS.png'),
                      ('跨Train/Test分支画像','figures/TRAIN_TEST_BRANCH_PHENOTYPES.png'),
                      ('评分、安全与互补性','outcomes/OUTCOME_PATTERNS.png'),
                      ('Train真实场景','cases/TRAIN_QUALITATIVE_CASES.png'),
                      ('Test真实场景','cases/TEST_QUALITATIVE_CASES.png')]:
        sections.append(f'<h2>{html.escape(title)}</h2><img src="{rel}">')
    cases=jread(ROOT/'cases/CASES.json')
    body='<h1>ReCogDrive 修复规律证据浏览</h1><p>完整报告见 <a href="REPAIR_PATTERN_REPORT.md">REPAIR_PATTERN_REPORT.md</a>。纯 CPU 分析；所有评分来自既有冻结实验。</p>'
    body+=''.join(sections)+'<h2>案例来源与全部候选均分</h2>'
    for c in cases:
        body+=f'<details><summary>{c["population"]} E{c["choice"]} {c["token"]}：{c["Base_PDMS"]:.2f} → {c["teacher_PDMS"]:.2f}</summary><pre>{html.escape(json.dumps(c,ensure_ascii=False,indent=2))}</pre></details>'
    (ROOT/'EVIDENCE_BROWSER.html').write_text('<!doctype html><meta charset="utf-8"><title>ReCogDrive修复规律</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;line-height:1.7;color:#172638}img{max-width:100%;border:1px solid #ddd}pre{white-space:pre-wrap;font-size:12px}details{border-bottom:1px solid #ddd;padding:8px}</style>'+body)


def independent_checks():
    checks={}
    for pop,expected in [('train',85813),('test',12146)]:
        with np.load(N/'requests'/pop/'MATRIX.npz',allow_pickle=False) as z:scores,indices=z['scores'],z['indices']
        with np.load(N/'frozen_teacher/selections'/f'{pop}.npz',allow_pickle=False) as z:
            q,choice,tokens=z['q_points'],z['choice'],z['tokens']
            assert np.array_equal(indices,z['indices'])
        assert len(tokens)==expected and len(np.unique(tokens))==expected
        assert np.array_equal(q,scores[:,:,:,0].mean(2)*100)
        assert np.array_equal(choice,q.argmax(1))
        with np.load(ROOT/'scene_metadata'/f'{pop}_metadata.npz',allow_pickle=False) as z:
            assert np.array_equal(tokens,z['token'])
        summary=jread(N/'frozen_teacher'/f'{pop.upper()}_EVALUATION.json')
        assert abs(q[:,0].mean()-summary['Base_PDMS_points'])<1e-10
        assert abs(q.max(1).mean()-summary['full_teacher_PDMS_points'])<1e-10
        rows=[r for r in read_csv(ROOT/'outcomes/BRANCH_PROFILES.csv') if r['population']==pop]
        assert sum(int(r['n']) for r in rows)==expected
        assert abs(sum(float(r['whole_population_gain_contribution']) for r in rows)-(q.max(1)-q[:,0]).mean())<1e-10
        checks[pop]=dict(scenes=expected,selection_source_exact=True,metadata_token_order_exact=True,
                         Base_PDMS=float(q[:,0].mean()),teacher_PDMS=float(q.max(1).mean()),
                         branch_contributions_add_exactly=True)
    assert digest(N/'frozen_teacher/BUNDLE.json')=='ed0a1bef8b27d89e7bb4eb8f32c93134204d66c42184ddecaa1831c5a7188dfc'
    assert digest(N/'frozen_teacher/library.json')=='4b04bc28a237f83bf35dc5a175e2affbfacfaff67af691c02b3c9cbc5d4f8b44'
    jwrite(ROOT/'NUMERIC_CHECKS.json',dict(status='PASS',checks=checks,frozen_bundle_unchanged=True,
            frozen_library_unchanged=True,model_or_scoring_executed=False,
            trajectories_verification='85813 original Train per-scene NPZ verified by stored SHA, see trajectories/TRAIN_EXTRACTION.json',
            outcome_arithmetic_verification='See outcomes/VERIFICATION.json for exact metric/Shapley closure'))


def package():
    archive=ROOT/'REPAIR_PATTERN_EVIDENCE_COMPACT.zip'
    excludes={'train_metadata.csv','test_metadata.csv','TRAIN_SCENE_GEOMETRY.csv','TEST_SCENE_GEOMETRY.csv',
              'FIT_PAIR_BASE_PLANNING_ERRORS.csv'}
    selected=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.zip','.pyc','.log','.npz'] or p.name in excludes or p.name in ['MANIFEST.json','COMPLETE.json','ANALYSIS_STDOUT.txt'] and p.parent==ROOT:
            continue
        if p.stat().st_size>12_000_000:continue
        selected.append(p)
    jwrite(ROOT/'MANIFEST.json',dict(schema='recogdrive_repair_patterns_deliverable_v1',status='COMPLETE',
               scope='offline CPU analysis; all original assets preserved',
               compact_archive_omits='large NPZ and full per-scene CSV; available at original local paths',
               files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in selected]))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected+[ROOT/'MANIFEST.json']:z.write(p,p.relative_to(ROOT))
    jwrite(ROOT/'COMPLETE.json',dict(status='COMPLETE',scope='full Train/Test repair pattern discovery',
             report=str(ROOT/'REPAIR_PATTERN_REPORT.md'),archive=str(archive),archive_bytes=archive.stat().st_size,
             archive_sha256=digest(archive),manifest_sha256=digest(ROOT/'MANIFEST.json'),
             no_training_resume=True,new_gpu_jobs=0,remote_hosts_used=[]))
    print(json.dumps(jread(ROOT/'COMPLETE.json')))


def main():
    make_figures()
    qualitative()
    independent_checks()
    report()
    package()


if __name__=='__main__':main()
