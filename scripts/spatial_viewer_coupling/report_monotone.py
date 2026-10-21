"""Source-linked corrected report; refuses to render an unsealed evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import contracts as ct, audit
from .report_results import table

COMPARATORS = ['baseline', 'molecular_quality_balanced', 'spatial_quality_balanced',
               'calicost_quality_balanced', 'integrated_quality_balanced', 'consensus_quality_balanced']
DISPLAY = ['smooth_coverage', 'rank_coverage', 'spline_coverage', *COMPARATORS,
           'coverage_Coff_k0', 'coverage_Coff_k3', 'coverage_Coff_k3_matched_burden',
           'coverage_Con_k3', 'phase_Con_k3', 'none_Coff_k3',
           'coverage_Coff_prioroff_k3', 'coverage_Coff_identity_k3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation-id', default='evaluation_v2_monotone')
    args = ap.parse_args()
    out, ev = ct.OUT, ct.OUT/args.evaluation_id
    m = json.loads((ev/'evaluation_manifest.json').read_text())
    if ct.sha(ev/'evaluation_manifest.json') != (ev/'COMPLETE.sha256').read_text().strip():
        raise ValueError('evaluation seal mismatch')
    for name, digest in m['files'].items():
        if ct.sha(ev/name) != digest:
            raise ValueError('evaluation payload drift: '+name)
    audit.check_records(m['inputs']+m['source_code'])
    archive = out/'report_v1_archive'
    if not (archive/'RESULTS.md').is_file() or not (archive/'report_manifest.json').is_file():
        raise ValueError('original report must be archived before replacing the current report')
    old = out/m['source_evaluation_id']
    head = pd.read_csv(ev/'headline.csv')
    p6 = head[head.section.eq('P6_rep1')]
    correlations = pd.read_csv(ev/'P6_rep1_coverage_correlations.csv')
    rho = correlations[correlations.stage.eq('native_score') & correlations.scope.eq('all_spots')].set_index('model')
    rawrho = correlations[correlations.stage.eq('raw') & correlations.scope.eq('all_spots')].set_index('model')
    summary = p6.pivot(index='model', columns='mode', values='f1').reindex(DISPLAY)
    summary['rho_vs_coverage'] = rho.spearman_vs_smooth_coverage
    summary['raw_detector_rho_vs_coverage'] = rawrho.spearman_vs_smooth_coverage
    summary = summary[['raw', 'coverage_residual', 'rho_vs_coverage', 'raw_detector_rho_vs_coverage']].reset_index()
    summary.columns.name = None
    summary.to_csv(out/'headline_p6.csv', index=False)
    pair = pd.read_csv(ev/'P6_rep1_paired_spatial.csv')
    primary = pair[pair.model.eq('coverage_Coff_k3')]
    comp = pair[pair.model.isin(COMPARATORS)].copy()
    comp.to_csv(out/'comparator_vs_monotone_null.csv', index=False)
    grid = pd.read_csv(ev/'grid_sensitivity.csv')
    cg = grid[grid.section.eq('P6_rep1') & grid.model.isin(COMPARATORS)].copy()
    cg.to_csv(out/'comparator_grid_sensitivity.csv', index=False)
    interaction = pd.read_csv(ev/'context_interaction.csv')
    agreement = pd.read_csv(ev/'P6_rep1_rank_null_agreement.csv')
    k0_agree = agreement[agreement.model.eq('coverage_Coff_k0')]
    smoothed = summary.set_index('model')
    diagnostic = rho.loc[[f'coverage_Coff_k{k}' for k in range(4)],
        ['spearman_vs_smooth_coverage', 'pearson_vs_smooth_coverage', 'max_abs_rank_difference']].reset_index()
    diagnostic.to_csv(out/'monotone_iteration_diagnostics.csv', index=False)
    positive = comp[comp.delta_f1 > 0]
    assert rho.loc['coverage_Coff_k0', 'max_abs_rank_difference'] == 0
    assert np.isclose(rho.loc['coverage_Coff_k0', 'spearman_vs_smooth_coverage'], 1.)
    comparator_statement = ('六个既有 comparator 在固定 detector 的 raw 和 residual 两个端点，均未胜过 rank 或 spline null（24 个配对点估计均不为正）。'
                           if positive.empty else '固定 detector 下有正点估计的 comparator 比较：'+str(positive[['model', 'mode', 'reference', 'delta_f1']].to_dict('records'))+'；区间和置换 p 见下述 CSV。')
    primary_success = bool((primary.ci_low > 0).all()) and len(primary) == 4
    primary_statement = ('主回路 K3 在固定 detector 的 raw/residual 两个端点均有相对两个 null 的正区间。'
                         if primary_success else '主回路 K3 未胜过 monotone-coverage null，当前回路仍应 **DROP 作为精度升级的主张**。')
    source = lambda name: f'[{name}]({args.evaluation_id}/{name})'
    oldsource = lambda name: f'[{name}]({m["source_evaluation_id"]}/{name})'
    compact = summary[['model', 'raw', 'coverage_residual', 'rho_vs_coverage']]
    comp_delta = comp.pivot(index='model', columns=['mode', 'reference'], values='delta_f1').reindex(COMPARATORS)
    comp_delta.columns = [' / '.join(c) for c in comp_delta.columns]
    comp_delta = comp_delta.reset_index()
    grid_fractions = cg.pivot(index='model', columns=['mode', 'reference'], values='fraction_positive').reindex(COMPARATORS)
    grid_fractions.columns = [' / '.join(c) for c in grid_fractions.columns]
    primary_grid = grid[grid.section.eq('P6_rep1') & grid.model.eq('coverage_Coff_k3')]
    raw_wins = cg[(cg['mode'] == 'raw') & (cg.reference == 'rank_coverage')].set_index('model')
    residual_wins = cg[(cg['mode'] == 'coverage_residual') & (cg.reference == 'rank_coverage')].set_index('model')
    nwin = lambda row: int(round(row.fraction_positive*row.n_grid))
    raw_sentence = 'raw 网格中，'+ '、'.join(f'{name}: {nwin(raw_wins.loc[name])}/81' for name in COMPARATORS)+ ' 个同设置比较胜过 rank null；spline null 的 raw 计数相同。'
    spline_residual = cg[(cg['mode'] == 'coverage_residual') & (cg.reference == 'spline_coverage')]
    residual_sentence = 'residual 网格中，'+ '、'.join(f'{name}: {nwin(residual_wins.loc[name])}/81' for name in COMPARATORS)+ ' 胜过 rank null；对 spline null 的六个 comparator 总正点数为 '+str(sum(nwin(row) for row in spline_residual.itertuples()))+'。这些网格点高度相关，计数不是独立重复试验。'
    legacy_head = pd.read_csv(old/'headline.csv')
    old_coverage = float(legacy_head[(legacy_head.section == 'P6_rep1') & (legacy_head.model == 'smooth_coverage') & (legacy_head['mode'] == 'coverage_residual')].f1.iloc[0])
    corrected_coverage = float(smoothed.loc['smooth_coverage', 'coverage_residual'])
    text = [
        '# RESULTS — monotone-coverage null 校正（2026-09-16）\n\n'+primary_statement+' '+comparator_statement,
        '**撤回旧 headline 的 residual 增益作为 coverage-independent 证据的解释。** 线性残差化不对分数的单调变换保持不变；coverage 的非线性表示本身即可产生高 residual F1。本报告所有 delta、CI 和 permutation p 均以明确列出的 monotone null 为参照，不再以未校正的 coverage residual 差值支持精度主张。原报告原样归档于 [report_v1_archive/RESULTS.md](report_v1_archive/RESULTS.md)，原 evaluation_v1 与推断 bundle 均未改动。',
        '## P6 headline\n\n固定 detector：seed=90、grow=80、min-size=5、split=0.4。F1 来源 '+source('headline.csv')+'；rho 是进入 detector 处理前的原生分数对 smoothed coverage 的 Spearman，来源 '+source('P6_rep1_coverage_correlations.csv')+'。可下载 [headline_p6.csv](headline_p6.csv)，其中还含 raw detector 分数的 rho。\n\n'+table(compact),
        '**k0 prior 是 smoothed coverage 的单调变换，Spearman=1.000000**。其最大秩差为 '+f'{rho.loc["coverage_Coff_k0", "max_abs_rank_difference"]:.1f}'+'；它与 rank null 的默认 residual mask '+('逐位相同' if bool(k0_agree[(k0_agree.seed == 90) & (k0_agree.grow == 80) & (k0_agree.min_size == 5) & (k0_agree["split"] == .4) & (k0_agree["mode"] == "coverage_residual")].masks_exact_equal.all()) else '并非逐位相同')+f'。全部 raw/residual 网格中 {int(k0_agree.masks_exact_equal.sum())}/{len(k0_agree)} 个 mask 逐位相同。来源 '+source('P6_rep1_rank_null_agreement.csv')+'。\n\n迭代只轻微改变 coverage 的秩；Pearson 的变化不能替代此诊断。来源 '+source('P6_rep1_coverage_correlations.csv')+'；导出 [monotone_iteration_diagnostics.csv](monotone_iteration_diagnostics.csv)。\n\n'+table(diagnostic),
        '## 变换链与 null 定义\n\n记 H 为既有 viewer 的一阶图平滑，R(x) 为对 [1, raw UMI] 的 OLS 残差。posterior、smooth_coverage、rank_coverage、spline_coverage 全部使用 **raw=x；residual=H(R(x))**。burden comparator 保留已注册定义 **raw=H(burden)；residual=H(R(burden))**。因此 burden 的原生分数和平滑后分数分开记录，不隐瞒这一步差异。每一 detector 行均携带变换链；完整定义见 '+source('P6_rep1_transform_chains.csv')+'。',
        'rank_coverage=rankdata(H(UMI))。spline_coverage 是 log1p(H(UMI)) 上的单调 PCHIP：固定分位概率 (0,.10,.25,.50,.75,.90,1) 对应 coverage 自身的分位数节点，映射到相应概率；重复节点按概率均值合并。节点仅由 coverage 决定，不读 GT，不择优。来源 '+source('P6_rep1_spline_knots.json')+'；[校正方案](../../scripts/spatial_viewer_coupling/MONOTONE_NULL_PROTOCOL.md)。两种 null 分别报告；不按 F1 选参照。',
        f'平滑不对称修正后，smooth_coverage residual F1={corrected_coverage:.6f}（旧值 {old_coverage:.6f}，仅用于错误定位，不作为主效应检验）。rank null residual F1={smoothed.loc["rank_coverage", "coverage_residual"]:.6f}，解释了 k0 的 residual 表现。来源 '+source('headline.csv')+' 与 '+oldsource('headline.csv')+'。',
        '## 相对 null 的配对空间统计\n\n下表全部为主回路 K3 减去该行 reference。95% CI：2,000 次配对空间 block bootstrap；p：999 次空间块置换的单侧“有提升”检验。双侧 p 和全部模型比较在 '+source('P6_rep1_paired_spatial.csv')+'；不做多重性校正，不将 exploratory p 当作 familywise 结论。\n\n'+table(primary[['mode', 'reference', 'delta_f1', 'ci_low', 'ci_high', 'spatial_null_p', 'spatial_null_two_sided_p']]),
        '## 既有六个 comparator 是否胜过 null\n\n'+comparator_statement+' 这里是同一默认 detector 的事实比较，不修改 `report/09_18_record.md` §7.5 的既有结论。下表为 comparator−null 的 F1；每个差值的 CI、置换 p 和胜出概率均在 [comparator_vs_monotone_null.csv](comparator_vs_monotone_null.csv)，上游来源 '+source('P6_rep1_paired_spatial.csv')+'。\n\n'+table(comp_delta),
        '完整 81 点网格并不等于默认 detector：以下是各 comparator 在对应同一网格设置胜过 null 的比例（严格 delta>0）。网格比较与固定 detector 的主端点分别报告，不选择最佳 detector。来源 '+source('grid_sensitivity.csv')+' 与 [comparator_grid_sensitivity.csv](comparator_grid_sensitivity.csv)。\n\n'+table(grid_fractions.reset_index()),
        raw_sentence+' '+residual_sentence+' 来源 '+source('grid_sensitivity.csv')+'。',
        '主回路在全网格的敏感性：来源 '+source('grid_sensitivity.csv')+'；逐点差值见 '+source('grid_contrasts.csv')+'。\n\n'+table(primary_grid[['mode', 'reference', 'n_grid', 'min_delta', 'median_delta', 'max_delta', 'fraction_positive']]),
        '## Context、replicate、WES 与 QC\n\n正式 model×context 为 P6−平均(P4,DCIS) 的模型减 null 差值。所有共享模型都已重算；主回路如下。P4 使用 copyKAT silver，P6/DCIS 使用 pathology，因此仅为当前 section 条件下比较，不能作为 donor 层面的交互结论。来源 '+source('context_interaction.csv')+'。\n\n'+table(interaction[interaction.model.eq('coverage_Coff_k3')][['endpoint', 'reference', 'interaction', 'ci_low', 'ci_high', 'two_sided_bootstrap_p']]),
        '本轮没有修改 selected calls、posterior 或 UMI，故 replicate Jaccard、callable-WES Wilson CI、intrinsic QC、burden–UMI 不重算。它们仍可追溯到 '+oldsource('replicate_jaccard.csv')+'、'+oldsource('patient_union_wes.csv')+' 及各 section 的 `*_intrinsic_qc.csv`、`*_burden_umi.csv`。P6 WES eligible n=5，只作支持性证据；frozen molecular posterior 已含跨 replicate recurrence，因此 Jaccard 不是独立 replicate 验证。P6_rep2 没有 pathology，F1 留空。DCIS 缺少历史 QB/multistate，不能伪造 comparator。',
        '## 冻结、重放与适用范围\n\n原有每次迭代的 true/complement/permuted 推断重放记录已核验哈希并继承；本次不把它们声称为重新运行的推断。新增 coverage-null 与所有分数处理在三个标签上下文中重放，数值哈希相同，数值阶段禁止文件 I/O；具体数量及输入哈希见各 section 的 `*_manifest.json`、`*_processing_label_swap.csv` 和 '+source('evaluation_manifest.json')+'。所有 native/processed scores、valid masks、默认预测与全网格 packed masks 均已冻结，可直接审计。',
        '预注册保留：§7.5 的 detector 网格、默认 detector、raw/residual、P6 主分析及空间重采样方法；模型参数和 K 未改变。**Post hoc**：依据用户复核补充两个 monotone null、修正 coverage 残差后平滑、补齐中间迭代全网格、增加秩诊断并更换所有空间精度比较的参照。这是纠错与加强对照，不能追称为原先预注册。新方案先写入 MONOTONE_NULL_PROTOCOL.md，再运行校正评估；所有节点和变换均不由 GT 拟合。',
        '默认入口 `evaluate_registered.slurm` 已转向 `evaluate_monotone.py`。未来换 Beagle 输入用新的 run/evaluation ID：若缺少对应缓存，入口先生成冻结评分缓存，再强制加入两个 null 并校验 run ID、输入和代码哈希。旧 evaluate.py 仅保留为旧结果的来源与缓存生成器；生产 filter、Beagle、已有 viewer study/profile 和 §7.5 均未改动。',
        '## 机制贡献与建议\n\n**KEEP 契约、显式有界通道、冻结重放、可测量的循环性；DROP 当前回路的精度升级主张。** A 保留五态后验、information×reliability、QC 和稀疏 REF/ALT；B 明示 prior 来源、graph、membership 与 GT-free 断言。通道和有效观测数封顶，prior-off/identity-destroyed 消融及标签交换让反馈的影响可测量。这些是接口与可检验性成果，与精度增益彻底分开。',
        '## 未解决问题\n\n1. 对 raw UMI 做线性残差化无法消除任意非线性 coverage 信号；胜过这两个特定 null 也不足以证明 coverage independence。\n2. K=3 是固定有限迭代点，并非已数值收敛；弱反馈与覆盖先验主导在当前参数下并未带来可靠精度收益。\n3. 空间置换保留等大小块内标签结构，但会打断跨块 cluster；尾部不足一块的 spots 固定。bootstrap 条件于单 section，多个未校正比较与异质 context 标签限制推广。\n4. frozen posterior、phase/CNV 的来源及 replicate recurrence 限制独立性；新的 Beagle 输入能否改变结论必须重新运行，不能推断。\n5. 现有 viewer GUI 尚不显示 A 的所有五态/QC 字段；本次 null 仅在评估产物中导出，不改既有 viewer profile。'
    ]
    (out/'RESULTS.md').write_text('\n\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = ['smooth_coverage', 'rank_coverage', 'spline_coverage', 'baseline',
            'consensus_quality_balanced', 'coverage_Coff_k0', 'coverage_Coff_k3']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, mode in zip(axes, ['raw', 'coverage_residual']):
        values = smoothed.reindex(show)[mode]
        ax.barh(np.arange(len(show)), values, color=['#aaa', '#4a8b65', '#70a285', '#888', '#7897ad', '#b3ad8a', '#c97849'])
        ax.set_yticks(np.arange(len(show)), show)
        ax.invert_yaxis()
        ax.set(xlabel='P6 pathology F1', title=mode, xlim=(0, .55))
        ax.grid(axis='x', alpha=.2)
        for i, x in enumerate(values):
            ax.text(x+.006, i, f'{x:.3f}', va='center', fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'p6_fixed_detector.{ext}', dpi=180)
    plt.close(fig)
    files = ['RESULTS.md', 'headline_p6.csv', 'p6_fixed_detector.png', 'p6_fixed_detector.pdf',
             'comparator_vs_monotone_null.csv', 'comparator_grid_sensitivity.csv', 'monotone_iteration_diagnostics.csv']
    report_manifest = dict(schema='monotone_null_report_v2', primary_pass=primary_success,
        inputs=[ct.entry(ev/'evaluation_manifest.json', 'corrected_evaluation'),
                ct.entry(Path(__file__), 'report_code'),
                ct.entry(ct.CODE/'report_results.py', 'legacy_table_renderer'),
                ct.entry(archive/'report_manifest.json', 'archived_report_manifest')],
        files={name: ct.sha(out/name) for name in files},
        comparator_positive_default_contrasts=len(positive), comparator_total_default_contrasts=len(comp))
    (out/'report_manifest.json').write_text(json.dumps(report_manifest, indent=2, sort_keys=True)+'\n')
    print(out/'RESULTS.md')


if __name__ == '__main__':
    main()
