"""Render conclusions and source-linked tables from frozen evaluation, without fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from . import contracts as ct


def table(frame):
    headers=list(frame.columns); lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-id',default='evaluation_v1');ap.add_argument('--run-id',default='registered_v1');a=ap.parse_args()
    out=ct.OUT; ev=out/a.evaluation_id; run=out/a.run_id
    if (ev/'COMPLETE.sha256').read_text().strip()!=ct.sha(ev/'evaluation_manifest.json'):raise ValueError('evaluation seal')
    em=json.loads((ev/'evaluation_manifest.json').read_text())
    for name,digest in em['files'].items():
        if ct.sha(ev/name)!=digest:raise ValueError(f'evaluation file drift {name}')
    head=pd.read_csv(ev/'headline.csv'); p6=head[head.section.eq('P6_rep1')]
    models=['smooth_coverage','baseline','molecular_quality_balanced','spatial_quality_balanced','calicost_quality_balanced','integrated_quality_balanced','consensus_quality_balanced','coverage_Coff_k0','coverage_Coff_k3','coverage_Coff_k3_matched_burden','coverage_Con_k3','phase_Con_k3','none_Coff_k3','coverage_Coff_prioroff_k3','coverage_Coff_identity_k3']
    headline=p6[p6.model.isin(models)].pivot(index='model',columns='mode',values='f1').reindex(models).reset_index(); headline.columns.name=None
    headline.to_csv(out/'headline_p6.csv',index=False)
    pair=pd.read_csv(ev/'P6_rep1_paired_spatial.csv'); primary=pair[pair.model.eq('coverage_Coff_k3')&pair.reference.eq('smooth_coverage')]
    success=len(primary)==2 and bool((primary.ci_low>0).all())
    rep=pd.read_csv(ev/'replicate_jaccard.csv'); wes=pd.read_csv(ev/'patient_union_wes.csv'); interaction=pd.read_csv(ev/'context_interaction.csv'); grid=pd.read_csv(ev/'grid_sensitivity.csv')
    swaps=pd.concat([pd.read_csv(ev/f'{sec}_label_swap.csv') for sec in em['sections']],ignore_index=True)
    diagnostics=[]
    for p in sorted(run.glob('*/*/run_manifest.json')):
        m=json.loads(p.read_text())
        for d in m['iterations']:diagnostics.append(dict(section=m['section'],arm=m['arm'],**d,source=str(p.relative_to(out))))
    diag=pd.DataFrame(diagnostics);diag.to_csv(out/'iteration_diagnostics.csv',index=False)
    local=diag[diag.arm.eq('coverage_Coff')&diag.iteration.eq(3)]
    support=rep.merge(wes,on=['patient','model']);support=support[support.model.isin(models)][['model','intersection','union','jaccard','eligible_n','hits','recall','ci_low','ci_high']]
    sources=lambda f:f'[{f}]({a.evaluation_id}/{f})'
    conclusion='P6 在本次冻结 detector 下同时赢过平滑 coverage 的 raw 与 residual F1，且两个配对空间 bootstrap 区间下界均为正。' if success else 'P6 没有满足预先声明的“同时赢过平滑 coverage 的 raw 与 residual F1，且配对空间 bootstrap 区间均支持正增益”的标准。'
    ablation=pair[pair.model.eq('coverage_Coff_k3')&pair.reference.isin(['coverage_Coff_k0','coverage_Coff_prioroff_k3','coverage_Coff_identity_k3'])]
    wide=headline.set_index('model')
    interpretation=(f"原始 F1：K3={wide.loc['coverage_Coff_k3','raw']:.6f}，平滑 coverage={wide.loc['smooth_coverage','raw']:.6f}。"
       f" residual F1：K3={wide.loc['coverage_Coff_k3','coverage_residual']:.6f}，coverage={wide.loc['smooth_coverage','coverage_residual']:.6f}，但 iteration0={wide.loc['coverage_Coff_k0','coverage_residual']:.6f}、prior-off={wide.loc['coverage_Coff_prioroff_k3','coverage_residual']:.6f}、identity-destroyed={wide.loc['coverage_Coff_identity_k3','coverage_residual']:.6f}。"
       "残差优势在无回路的 coverage-rank anchor 中已存在，不能归因于反馈或变异身份。两个 coverage 表示经历了不同的非线性变换/平滑，故跨表示 residual 差异不能被直接写成新增分子信号。")
    text=[f'# RESULTS — spatial filter ↔ SPARCALviewer coupling\n\n{conclusion}',
      interpretation+' 来源：'+sources('headline.csv')+'、'+sources('P6_rep1_paired_spatial.csv')+'。',
      '建议：**KEEP 类型化证据契约、显式有界通道、冻结与重放机制；DROP 当前参数回路作为 production 精度升级的推广主张**。' if not success else '建议：**KEEP 契约与本次候选用于后续独立验证；本次不覆盖 production**。',
      '## P6 主结果\n\n固定 detector：seed 90 / grow 80 / min-size 5 / split 0.4。`coverage_Coff_k3` 是预先声明的主回路；`_matched_burden` 用其匹配 call set 的 ALT-positive burden 重做历史 detector。其他回路不通过 GT 选优。\n\n来源：'+sources('headline.csv')+'；精简表 [headline_p6.csv](headline_p6.csv)。\n\n'+table(headline),
      '## 配对空间不确定性\n\n主对照为 **平滑 coverage**；以下每行的 delta 都是 K3−coverage。block bootstrap 与空间 block permutation 的完整设定见 DESIGN_IMPLEMENTATION_NOTES。来源：'+sources('P6_rep1_paired_spatial.csv')+'。\n\n'+table(primary[['mode','delta_f1','ci_low','ci_high','win_probability','spatial_null_p']]),
      '机制归因的直接消融（同一冻结 detector）：来源 '+sources('P6_rep1_paired_spatial.csv')+'。\n\n'+table(ablation[['mode','reference','delta_f1','ci_low','ci_high','win_probability']]),
      '固定网格的全体结果用于敏感性分析，不挑最优点。来源：'+sources('grid_sensitivity.csv')+'。\n\n'+table(grid[grid.section.eq('P6_rep1')&grid.reference.isin(['smooth_coverage','coverage_Coff_k0'])][['mode','reference','n_grid','min_delta','median_delta','max_delta','fraction_positive']]),
      '## Replicate 与 WES silver\n\n来源：'+sources('replicate_jaccard.csv')+'、'+sources('patient_union_wes.csv')+'。WES 分母为匹配候选 universe 中可调用的患者并集；区间为 Wilson 95% CI，不能据小分母建立独立精度结论。冻结 molecular posterior 已包含跨 replicate recurrence 信息（原 score_models.py 的 add_recurrence / molecular_quality），因此 Jaccard 是条件于这些输入的支持指标，不是独立 replicate 验证。\n\n'+table(support),
      '## 正式 context interaction\n\n效应为 P6−平均(P4,DCIS)。F1 中 P4 使用 copyKAT silver，P6/DCIS 使用 pathology，因此是当前 section 条件下的 interaction，不能外推成 donor-level 生物学交互。未使用“一个显著、另一个不显著”的推断。来源：'+sources('context_interaction.csv')+'。\n\n'+table(interaction[['endpoint','reference','interaction','ci_low','ci_high','two_sided_bootstrap_p']]),
      '## 回路与泄漏审计\n\n'+f'完成 {len(diag)} 个冻结迭代，执行 {len(swaps)} 个 true/complement/permuted 标签上下文重放，全部数值哈希一致：{bool(swaps.pass_exact.all())}。所有数值重放禁止文件 I/O，并用真实 WES 路径的读取拒绝测试证明护栏生效。来源：各 section 的 `*_label_swap.csv` 与 `*_manifest.json`。原 inference manifest 的 pending 字段保留历史；评价完成状态在独立 evaluator manifest 中。',
      'K=3 是固定评估点，不等于已收敛。来源：[iteration_diagnostics.csv](iteration_diagnostics.csv) 及该表逐行列明的原始 run manifest。\n\n'+table(local[['section','prior_sup_change','fixed_point_error_bound','converged','selected_n','max_abs_feedback']]),
      '序列化 A/B 恢复更新的真实 P4 校验见 [exchange_smoke/final_result.json](exchange_smoke/final_result.json)。新 viewer studies 位于 [viewer_studies/](viewer_studies/)；每个 export manifest 记录 posterior 分区及切换解释 component 后的 membership 一致性。',
      '## QC、输入范围和可审计性\n\n各 section 的 `*_intrinsic_qc.csv` 给出 xMHC、DP<3、editing、dense-10bp、singleton；`*_burden_umi.csv` 同时给出匹配 burden–UMI 和 posterior score–UMI。这些指标与 WES/空间结果一起保留，不按结论选择性省略。\n\n'+ '\n'.join(f'- {sec}: '+sources(f'{sec}_intrinsic_qc.csv')+'；'+sources(f'{sec}_burden_umi.csv')+'；'+sources(f'{sec}_manifest.json') for sec in em['sections']),
      '每个原始输入在 `registered_v1/<SECTION>/prepared/input_ledger.json` 记录 path+SHA256，完成运行时重新核验；各 A/B 的 manifest 钉住输入、代码、参数和载荷。P4/P6 精确 REF/ALT 来自既有 VCF，未将 `max(DP,sum(AD))-ALT` 冒充 REF。DCIS 缓存 AD 与 VCF 精确核对，同时恢复零-read record presence。',
      '## 预注册、实施冻结与 post hoc\n\n沿用 §7.5 的 primary P6、固定 detector 网格、raw/residual、空间重采样、replicate/WES/QC 和正式 interaction。新回路数学形式、lambda、damping、K、prior 来源和消融在本轮任何 GT 评价之前写入 DESIGN 并冻结；它们不是通过 GT 拟合的参数。\n\n实施阶段预先声明的差异：DCIS 缺少历史 multistate/QB，使用标明来源的 AD adapter；native phase 自身依赖 CalicoST；不规则组织使用等大小空间块置换近似 null；P4 的空间标签为 silver。详见 DESIGN_REVIEW 与 DESIGN_IMPLEMENTATION_NOTES。post hoc 仅为结果整理、图形展示和机制解释，没有根据标签追加调参、筛选 detector 或重新选 K。',
      '## 没有精度增益时仍成立的机制贡献\n\nA 保留五态概率、information×reliability、QC 与精确稀疏 REF/ALT，二值矩阵退回兼容视图。B 将 prior 来源、空间图、有效 spot 和 region 显式化。CalicoST 有可审计开关；反馈重评分从固定底座出发，单通道与有效观测数均封顶。固定权重下映射收缩，标签交换和 prior-off/identity-destroyed 控制让循环影响成为可测量对象。这些是接口与可检验性贡献，不等同于独立生物验证或概率校准。',
      '## 失败模式与未解决问题\n\n1. K=3 的有限步输出不保证达到数值固定点；误差界按最坏收缩系数给出，不将小 call-set 改变等同于 prior 收敛。\n2. coverage anchor 和 posterior 单调变换本身可能保留主要 extent 信号；必须同时看 iteration0、prior-off、identity-destroyed 与 matched-burden，不能把继承 coverage 的分区称为新的分子定位证据。\n3. per-spot somatic posterior 均值以 0.5 为中心转成 bounded logit increment，尚未校准成生成式 tumor-vs-normal likelihood；这不保证适合低 somatic burden。frozen posterior 的有界比值是 generalized evidence，未重新拟合完整 CN/purity likelihood；native phase/CNV/purity 同源，属于 in-sample descriptive evidence。Coff 关闭本轮直接 scoring/prior 的 CalicoST 通道，但保留 frozen baseline 的配额及 QC 预算，不能称为重跑整个上游后的完全无 CalicoST pipeline。\n4. 空间块 null 会打断跨块 cluster；bootstrap 条件于单个 section。P4 silver 与 pathology 不同，WES eligible 分母很小。\n5. DCIS 没有历史五态/QB，且 xMHC 区间为 hg38 chr6:25–34Mb，P4/P6 继承 chr6:28–34Mb；跨 context QC 不可误写为完全相同定义。\n6. 既有 viewer GUI 可消费冻结 posterior sidecar，但尚未提供所有五态/QC 字段的交互显示；它们可通过新 A reader 读取。\n7. 新 Beagle 输入必须以新 run-id 重跑，并比较新输入哈希；本报告不预测未来输入改善的效果。']
    report=out/'RESULTS.md'
    with report.open('x') as f:f.write('\n\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    show=['smooth_coverage','baseline','consensus_quality_balanced','coverage_Coff_k0','coverage_Coff_k3','coverage_Con_k3','phase_Con_k3','coverage_Coff_identity_k3']
    for ax,mode in zip(axes,['raw','coverage_residual']):
        values=p6[p6['mode'].eq(mode)].set_index('model').reindex(show).f1
        ax.barh(np.arange(len(show)),values,color=['#777777','#aaaaaa','#6285a8','#99b9cc','#d87f45','#ad6688','#6d996c','#bbac88']);ax.set_yticks(np.arange(len(show)),show);ax.invert_yaxis();ax.set_xlabel('P6 pathology F1');ax.set_title(mode);ax.set_xlim(0,max(.5,float(values.max())*1.15));ax.grid(axis='x',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(out/f'p6_fixed_detector.{ext}',dpi=180)
    plt.close(fig)
    ct.dump(out/'report_manifest.json',dict(inputs=[ct.entry(ev/'evaluation_manifest.json','sealed_evaluation'),ct.entry(Path(__file__),'report_code')],files={name:ct.sha(out/name) for name in ('RESULTS.md','headline_p6.csv','iteration_diagnostics.csv','p6_fixed_detector.png','p6_fixed_detector.pdf')},primary_pass=success))
    print(report)

if __name__=='__main__':main()
