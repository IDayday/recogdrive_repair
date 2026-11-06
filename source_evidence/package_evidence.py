"""Package complete derived analyses and relevant frozen numerical evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import zipfile

OUT=Path(__file__).resolve().parent
N=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027')
A=N/'engineering/repair_patterns_20261103'
TOP='recogdrive_repair_pattern_evidence'
ARCHIVE=OUT/'recogdrive_repair_pattern_analysis_full.zip'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    start=time.monotonic()
    files={}
    for path in sorted(A.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.zip','.pyc']:
            files['analysis/'+str(path.relative_to(A))]=path
    small=['BUNDLE.json','library.json','space.npz','teacher_library.py',
           'TRAIN_EVALUATION.json','TEST_EVALUATION.json','SELECTIONS.json',
           'LOCAL_TEACHER_RESULTS.json','LOCAL_TEACHER_TRAIN_TEST_REPORT.md']
    small += [f'slot{i:02d}.npz' for i in range(8)]
    for name in small:
        files['source_evidence/frozen_teacher/'+name]=N/'frozen_teacher'/name
    for pop in ['train','test']:
        files[f'source_evidence/selections/{pop}.npz']=N/'frozen_teacher/selections'/f'{pop}.npz'
        files[f'source_evidence/scoring/{pop}_MATRIX.npz']=N/'requests'/pop/'MATRIX.npz'
    for rel in ['ALLTRAIN/SPACE_AND_CLUSTER.pt','INNER/SPACE_AND_CLUSTER.pt',
                'ALLTRAIN/CLUSTER_TRANSFER.pt','ALLTRAIN/BANK.json',
                'FROZEN_INNER_SELECTION.json','LOCAL_PAIR_QUALIFICATION.json']:
        files['source_evidence/fitting/'+rel]=N/'fitting'/rel
    files['source_evidence/test_all_nine_CANDIDATES.npz']=N/'engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz'
    bundle=json.loads((N/'frozen_teacher/BUNDLE.json').read_text())
    for ref in bundle['original_math_sources']:
        path=Path(ref['path'])
        if not path.is_file() or digest(path)!=ref['sha256']:
            raise RuntimeError('Original mathematical source identity mismatch: '+str(path))
        key='source_evidence/original_math_sources/'+path.name
        if key in files and files[key]!=path:raise ValueError('Duplicate source basename')
        files[key]=path
    files['package_evidence.py']=Path(__file__)
    readme='''# ReCogDrive 表征修复规律：完整分析与证据包

建议阅读顺序：

1. [核心结论总结](SUMMARY.md)
2. [完整分析报告](analysis/REPAIR_PATTERN_REPORT.md)
3. [图表与案例浏览](analysis/EVIDENCE_BROWSER.html)，解压后用浏览器打开
4. [论文主张与证据边界](analysis/PAPER_CLAIMS.csv)

本包包含完整分析目录中的报告、总结、代码、PNG/PDF/SVG 图表、带原始前视图的案例、全量逐场景 CSV/NPZ 以及校验记录；不再省略之前紧凑包中的大表格或 NPZ。

## 覆盖范围

- 完整 Train：85,813 场景，130 个原始会话。
- 完整 Test：12,146 场景，44 个原始会话。
- 源拟合成功配对：1,654 个。
- 五组显式噪声 S0–S4，在场景内平均，不作为五个独立训练种子。

## 内容

- `analysis/representation/`：聚类几何、D48 能量、实际施加向量、源配对语义、聚类与最终选择的关系。
- `analysis/scene_metadata/`：完整原始场景派生元数据、逐会话 bootstrap、跨 Train/Test 画像、外生分层对照及全部来源 SHA。
- `analysis/trajectories/`：完整 Train Base／选中教师轨迹、逐场景几何与 GT 误差、源簇规划偏差、敏感性与置信区间。
- `analysis/outcomes/`：完整分支统计、安全恢复、近并列与噪声稳定性、子集消融、评分代数与人口组成分析。
- `analysis/figures/`、`analysis/cases/`：可导出图表和规则选择的真实定性案例。
- `source_evidence/`：原冻结教师定义和算子参数、Train/Test 原始九候选五噪声评分矩阵与固定选择、Test 九候选轨迹、拟合聚类状态和原数学源码。

## 编号和解释

Base=choice0；E1–E8=choice1–8=实现slot0–7。源拟合簇 C0–C7 指 ALLTRAIN 重拟合标签，不能与内层 cluster 字段或最终 choice 混用。

所有教师成绩是离线真实评分后选择的 Oracle 上限，不是已部署选择器或学生成绩。Test 已在历史开发中被检查，本轮属于探索性分析。选择后画像不能直接解释成路由规则或缺失能力的因果证明。

## 校验与复现边界

`PACKAGE_MANIFEST.json` 列出每个包内文件的 SHA256、字节数、原始路径；`SHA256SUMS.txt` 可用 `sha256sum -c SHA256SUMS.txt` 在本目录校验。

解压得到的 analysis/README.md、MANIFEST.json 和 COMPLETE.json 保留之前分析归档的历史语义，可能提到旧“紧凑包”；当前完整交付以本入口和 PACKAGE_MANIFEST.json 为准。

包内有本轮全部派生统计证据，以及上述相关冻结数值输入；不包含模型大权重、完整 NAVSIM 原始日志／图像／地图，也未打入 85,813 个 Train 场景所有九候选原始逐文件轨迹。完整 Train Base／选中教师轨迹已包含，完整 Test 九候选轨迹已包含。原始外部资产通过来源路径和 SHA 追踪。

分析脚本保留原工作区路径及依赖引用，重新执行时需要映射到包内对应证据，并配置原环境和未打包的外部资产；仅阅读报告、图表和统计无需模型或 GPU。为来源身份取证保留的 .pt 文件仅应在可信 Python/Torch 环境中读取。

本轮没有重新拟合、重训或重跑模型评分，也没有恢复已停止的蒸馏任务。
'''
    summary='''# 核心总结

## 最适合论文支撑的主线

修复方向包含运动情境与规划偏差结构；同一粗粒度场景下可能存在相反规划误差，因而需要不同干预。Train/Test 中重复出现的响应画像支持方向的跨场景复用，但聚类编号和场景语义不足以直接确定最优修复。

## 主要实测结果

1. **源聚类有情境共性。** C0/C7 偏高速直行，C1 偏低速起步，C2 富集刹停，C3/C5 都偏左转。导航指令 Cramér's V=0.430，速度 η²=0.319；不意味着八簇就是八种互斥能力。
2. **同左转可有不同误差。** C3/C5 的 GT 转角44.65°/43.40°接近，Base S0 航向误差−6.59°/+3.36°、沿 GT 终点误差−0.710/+2.054m。沿程差2.765m，log-bootstrap 95% CI[1.813,3.742]。速度与城市组成不同，属于描述性比较。
3. **最终选择不是源聚类标签。** 所属簇专家与最终 Oracle 精确一致仅24.67%，含并列36.09%；同1654场景直接用所属簇专家63.83分，Oracle91.11分。
4. **最终分支的跨split画像重复。** E2偏前方停驶车辆／GT刹停，E4富集右转，E7富集GT起步，E8偏较快、较稀疏交通。八分支使用率相关0.9986。
5. **画像不能直接变路由规则。** E4虽然选中人群富集右转，作用于全部右转命令场景却使Train/Test下降20.90/18.96 PDMS；任何单修复无选择地应用完整Test均低于Base。
6. **Test收益集中于少量严重错误。** 7.92%场景存在Base安全指标失败，贡献76.72%教师总增益。当前完整Train/Test Oracle为97.5298/96.5544，对应Base95.0721/90.6792。
7. **低维修复存在结构，但有构造因素。** D48占12.5%维度，捕获62.62%归一化源差分能量。六个固定向量覆盖约75%的已修复场景；实际选中shift前八主方向解释约93%未中心化能量。
8. **PDMS与人类GT误差可以分离。** 修复人群 GT-ADE Train .671→1.091m，Test .847→1.103m；87.33%/72.08%正PDMS修复同时让GT-ADE变差。不能直接称为恢复人类轨迹或纠正过度保守。
9. **保留负面证据。** DDC均值下降Train1.2503/Test1.0884个百分点，而其评分权重为0。最优对次优差>1仅25.01%/29.37%，应以regret/近优集合解释选择难度。

## 建议的论文表述

在冻结规划接口上，成功修复的表征差分具有低维能量集中，并与速度、转向及制动情境相关；同一运动情境内还包含不同的规划偏差。训练和测试中的响应画像重复、候选互补性与PDMS/GT误差的分离，表明该接口可进行有条件的规划行为干预，但尚不足以将方向簇解释成已确定缺失的语义能力模块。

详细统计人口、置信区间、混杂因素、反例、证据路径与复现方法见完整报告及 PAPER_CLAIMS.csv。
'''
    extras={'README.md':readme.encode(),'SUMMARY.md':summary.encode()}
    records=[]
    temp=ARCHIVE.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for name,path in sorted(files.items()):
            if not path.is_file():raise FileNotFoundError(path)
            before=path.stat()
            h=hashlib.sha256()
            with path.open('rb') as src,archive.open(TOP+'/'+name,'w',force_zip64=True) as dest:
                for block in iter(lambda:src.read(8<<20),b''):
                    h.update(block);dest.write(block)
            after=path.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise RuntimeError('Source changed while packaging: '+str(path))
            records.append(dict(path=name,source=str(path),bytes=before.st_size,sha256=h.hexdigest()))
        for name,content in extras.items():
            archive.writestr(TOP+'/'+name,content)
            records.append(dict(path=name,source='generated delivery entry',bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        manifest=dict(schema='recogdrive_full_repair_evidence_package_v1',
                      analysis_population=dict(train=85813,test=12146,fit_pairs=1654),
                      files=records,excluded=['model weights','full NAVSIM raw assets','all Train nine-candidate per-file trajectories','old nested zip archives','Python bytecode caches'],
                      original_assets_unchanged=True,new_model_or_GPU_jobs=0)
        manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        archive.writestr(TOP+'/PACKAGE_MANIFEST.json',manifest_bytes)
        sums=''.join(r['sha256']+'  '+r['path']+'\n' for r in records)
        sums+=hashlib.sha256(manifest_bytes).hexdigest()+'  PACKAGE_MANIFEST.json\n'
        archive.writestr(TOP+'/SHA256SUMS.txt',sums)
    print(json.dumps(dict(stage='PACKAGE_WRITTEN',files=len(records)+2,bytes=temp.stat().st_size)),flush=True)
    with zipfile.ZipFile(temp) as archive:
        assert len(archive.infolist())==len(records)+2
        for r in records:
            h=hashlib.sha256()
            with archive.open(TOP+'/'+r['path']) as f:
                for block in iter(lambda:f.read(8<<20),b''):h.update(block)
            if h.hexdigest()!=r['sha256']:raise ValueError('Archive member hash mismatch: '+r['path'])
        assert archive.read(TOP+'/PACKAGE_MANIFEST.json')==manifest_bytes
        assert archive.read(TOP+'/SHA256SUMS.txt').decode()==sums
    temp.replace(ARCHIVE)
    archive_sha=digest(ARCHIVE)
    (OUT/'SHA256SUMS.txt').write_text(archive_sha+'  '+ARCHIVE.name+'\n')
    receipt=dict(status='COMPLETE',archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,
                 sha256=archive_sha,files=len(records)+2,uncompressed_bytes=sum(r['bytes'] for r in records)+len(manifest_bytes)+len(sums.encode()),
                 verification='All archive members streamed and SHA256 verified; source inode/size/mtime stable during packaging; manifest and checksum lists exact',
                 original_analysis=str(A),elapsed_seconds=time.monotonic()-start,new_gpu_jobs=0)
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
