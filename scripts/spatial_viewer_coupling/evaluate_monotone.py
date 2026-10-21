"""Corrected post-freeze evaluator; the default public evaluation entry.

Consumes sealed legacy score caches, automatically making one for a fresh run.
GT is read only here/legacy evaluate.py, never by monotone_controls or inference.
"""
import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, rankdata
from . import contracts as ct, audit, model, evaluate as legacy
from . import monotone_controls as controls

NULLS = controls.NULLS
MODES = ('raw', 'coverage_residual')
GRID = list(itertools.product((85, 90, 95), (70, 80, 90), (3, 5, 10), (.3, .4, .5)))
DEFAULT = (90, 80, 5, .4)


def check_evaluation_records(records):
    """Hash check inside evaluator; GT paths require an evaluation-only role."""
    for rec in records:
        if set(rec) != {'path', 'sha256', 'role'}:
            raise ValueError('incomplete evaluation provenance')
        p = Path(rec['path'])
        if not p.is_absolute() or not rec['role'] or not re.fullmatch('[0-9a-f]{64}', rec['sha256']):
            raise ValueError('invalid evaluation provenance')
        is_gt = p.name == 'tumor_groups.csv' or p.name.startswith('STmut_Table_S1')
        if is_gt and not rec['role'].startswith('evaluation_only_'):
            raise ValueError('GT requires explicit evaluation-only provenance')
        if ct.sha(p) != rec['sha256']:
            raise ValueError(f'evaluation input hash mismatch: {p}')


def verify_cache(path, run_id):
    mp = path / 'evaluation_manifest.json'
    if not mp.is_file() or not (path / 'COMPLETE.sha256').is_file():
        raise ValueError('incomplete source evaluation; never reuse a partial cache')
    if ct.sha(mp) != (path / 'COMPLETE.sha256').read_text().strip():
        raise ValueError('source evaluation seal mismatch')
    m = json.loads(mp.read_text())
    if m['run_id'] != run_id or m['all_label_swap_tests_passed'] is not True:
        raise ValueError('wrong run ID or incomplete original label-swap checks')
    for name, digest in m['files'].items():
        if Path(name).name != name or ct.sha(path / name) != digest:
            raise ValueError(f'source payload drift: {name}')
    check_evaluation_records([m[k] for k in ('source', 'audit_code', 'design', 'truth_input')])
    return m


def permutation_indices(xy, n=999):
    _, pitch = legacy.block_ids(xy)
    tile = np.floor((xy-xy.min(0))/(4*pitch)).astype(int)
    order = np.lexsort((xy[:, 0], xy[:, 1], tile[:, 0], tile[:, 1]))
    nfull = len(order)//32
    inside = order[:nfull*32].reshape(nfull, 32)
    tail = order[nfull*32:]
    out = np.tile(np.arange(len(xy)), (n, 1))
    rng = np.random.default_rng(legacy.SEED)
    for ix in out:
        ix[inside.ravel()] = inside[rng.permutation(nfull)].ravel()
    return out, len(tail)


def null_f1(y, predictions, indices):
    """F1 for all bijectively permuted label fields; preserves missing labels."""
    permuted = y[indices]
    keep = np.isfinite(permuted).astype(float)
    pos = (np.nan_to_num(permuted) > 0).astype(float)
    matrix = np.column_stack(list(predictions.values())).astype(float)
    tp = pos @ matrix
    denom = pos.sum(1)[:, None] + keep @ matrix
    return np.divide(2*tp, denom, out=np.zeros_like(tp), where=denom > 0)


def paired_summary(y, predictions, xy):
    blocks, _ = legacy.block_ids(xy)
    boot, _ = legacy.bootstrap_counts(y, predictions, blocks)
    names = list(predictions)
    indices, tail = permutation_indices(xy)
    simulated = dict(zip(names, null_f1(y, predictions, indices).T))
    actual = {name: legacy.metrics(y, pred)['f1'] for name, pred in predictions.items()}
    rows = []
    for name in names:
        for reference in NULLS:
            d = boot[name]-boot[reference]
            delta = actual[name]-actual[reference]
            null = simulated[name]-simulated[reference]
            rows.append(dict(model=name, reference=reference, delta_f1=delta,
                ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                win_probability=float(np.mean(d > 0)),
                spatial_null_p=(1+int(np.sum(null >= delta)))/1000,
                spatial_null_two_sided_p=(1+int(np.sum(abs(null) >= abs(delta))))/1000,
                n_bootstrap=2000, n_permutation=999, n_blocks=int(blocks.max())+1,
                block_side_pitches=4, permutation='equal_32_spot_spatial_order_blocks_tail_fixed',
                null_fixed_tail=tail, null_mapping_collisions=0, multiple_testing_adjusted=False))
    return rows, boot


def correlation(x, c):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(c) == 0:
        return np.nan, np.nan
    return float(spearmanr(x, c).statistic), float(np.corrcoef(x, c)[0, 1])


def replay_processing(scores, umi, graph, labels):
    arrays, processed, spline = controls.prepare_scores(scores, umi, graph)
    def flatten(a, p):
        return {**{'native__'+k: x for k, x in a.items()},
                **{mode+'__'+name: x for name, zz in p.items() for mode, x in zz.items()}}
    expected = ct.array_hash(flatten(arrays, processed))
    try:
        with legacy.no_io():
            open(__file__, 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('numerical I/O guard failed')
    rng = np.random.default_rng(legacy.SEED)
    contexts = dict(true=labels, complement=1-labels, permuted=rng.permutation(labels))
    rows = []
    for name, context in contexts.items():
        with legacy.no_io():
            a, p, s = controls.prepare_scores(scores, umi, graph)
            digest = ct.array_hash(flatten(a, p))
        if digest != expected or s != spline:
            raise AssertionError('new transform label-context replay mismatch')
        rows.append(dict(label_context=name, context_sha256=ct.array_hash({'labels': context}),
                         numerical_sha256=digest, pass_exact=True, io_guard_pass=True,
                         scope='all_native_and_processed_scores; inference_not_rerun'))
    return arrays, processed, spline, rows


def evaluate_section(root, source, out):
    section = root.name
    print(f'loading {section}', flush=True)
    source_manifest = json.loads((source/f'{section}_manifest.json').read_text())
    check_evaluation_records(source_manifest['inputs'])
    z = ct.checked_npz(source/f'{section}_maps.npz')
    scores = {k.removeprefix('score__'): v for k, v in z.items() if k.startswith('score__')}
    barcode, xy, umi, y = [z[k] for k in ('barcode', 'xy', 'umi', 'labels')]
    current_y, gt_inputs, label_kind = legacy.labels(section, barcode)
    if not np.array_equal(y, current_y, equal_nan=True):
        raise ValueError('cached labels no longer match declared GT input')
    consumed = [ct.entry(source/f'{section}_maps.npz', 'sealed_native_score_cache'),
                ct.entry(source/f'{section}_manifest.json', 'source_section_manifest'),
                ct.entry(source/f'{section}_label_swap.csv', 'original_inference_label_replays')] + gt_inputs
    valids = {}
    graph = None
    for name, score in scores.items():
        if '_k' in name and not controls.is_burden(name):
            arm, k = name.rsplit('_k', 1)
            path = root/arm/f'iteration_{int(k):02d}'/'B'
            p, g, m = ct.read_b(path, barcode)
            if m['section'] != section or not np.array_equal(score, p['tumor_prior']):
                raise ValueError('score cache / B mismatch')
            if graph is not None and (graph != g).nnz:
                raise ValueError('graph drift between arms')
            graph = g
            valids[name] = p['valid']
            consumed.extend([ct.entry(path/'manifest.json', 'frozen_B'),
                             ct.entry(path/'prior.npz', 'valid_mask_and_prior'),
                             ct.entry(path/'graph.npz', 'frozen_graph')])
        else:
            valids[name] = np.ones(len(y), bool)
    if graph is None:
        raise ValueError('no frozen B graph')
    swaps = pd.read_csv(source/f'{section}_label_swap.csv')
    if not swaps.pass_exact.all() or not swaps.io_guard_pass.all():
        raise ValueError('original inference replay failure')
    complete, processed, spline, replays = replay_processing(scores, umi, graph, y)
    for name in NULLS:
        valids[name] = np.ones(len(y), bool)
    ct.dump(out/f'{section}_spline_knots.json', spline)
    pd.DataFrame(replays).assign(section=section).to_csv(out/f'{section}_processing_label_swap.csv', index=False)
    corr, chains = [], []
    coverage = complete['smooth_coverage']
    for name, native in complete.items():
        chain = controls.transform_chain(name)
        chains.append(dict(section=section, model=name, **chain))
        for stage, x in [('native_score', native), *processed[name].items()]:
            for scope, keep in [('all_spots', np.ones(len(y), bool)), ('valid_spots', valids[name])]:
                xx, cc = x[keep], coverage[keep]
                rho, pearson = correlation(xx, cc)
                corr.append(dict(section=section, model=name, stage=stage, scope=scope,
                    n_spots=int(keep.sum()), spearman_vs_smooth_coverage=rho,
                    pearson_vs_smooth_coverage=pearson,
                    max_abs_rank_difference=float(np.max(abs(rankdata(xx)-rankdata(cc)))) if len(xx) else np.nan,
                    **chain))
    pd.DataFrame(corr).to_csv(out/f'{section}_coverage_correlations.csv', index=False)
    pd.DataFrame(chains).to_csv(out/f'{section}_transform_chains.csv', index=False)
    rows, paired, agreement = [], [], []
    default_preds, boot_by_mode, all_predictions = {}, {}, []
    chain_by_name = {r['model']: {k: r[k] for k in ('native_score', 'raw_chain', 'residual_chain')} for r in chains}
    for mode in MODES:
        predictions = {}
        grid_masks = {}
        for name, zz in processed.items():
            masks = []
            for seed, grow, size, split in GRID:
                ids = model.regions(zz[mode], barcode, graph, valids[name], seed, grow, size, split)
                pred = ids >= 0
                masks.append(pred)
                all_predictions.append(np.packbits(pred))
                rows.append(dict(section=section, model=name, mode=mode, seed=seed, grow=grow,
                    min_size=size, split=split, n_regions=len(set(ids)-{-1}), predicted_n=int(pred.sum()),
                    label_kind=label_kind, **chain_by_name[name], **legacy.metrics(y, pred)))
                if (seed, grow, size, split) == DEFAULT:
                    predictions[name] = pred
            grid_masks[name] = np.array(masks)
        for name in ('smooth_coverage', 'spline_coverage', 'coverage_Coff_k0'):
            if name not in grid_masks:
                continue
            for j, (seed, grow, size, split) in enumerate(GRID):
                ndiff = int(np.sum(grid_masks[name][j] != grid_masks['rank_coverage'][j]))
                agreement.append(dict(section=section, mode=mode, model=name, reference='rank_coverage',
                    seed=seed, grow=grow, min_size=size, split=split, mask_mismatch_spots=ndiff,
                    masks_exact_equal=(ndiff == 0)))
        default_preds[mode] = predictions
        if np.isfinite(y).any():
            rr, bb = paired_summary(y, predictions, xy)
            paired.extend(dict(section=section, mode=mode, **r) for r in rr)
            boot_by_mode[mode] = bb
        print(f'{section} {mode}: {len(processed)*len(GRID)} detector cells', flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out/f'{section}_detector_grid.csv', index=False)
    pd.DataFrame(paired).to_csv(out/f'{section}_paired_spatial.csv', index=False)
    pd.DataFrame(agreement).to_csv(out/f'{section}_rank_null_agreement.csv', index=False)
    np.savez_compressed(out/f'{section}_grid_masks.npz', packed_masks=np.array(all_predictions),
                        n_spots=np.array(len(y)), barcode=barcode)
    np.savez_compressed(out/f'{section}_maps.npz', barcode=barcode, xy=xy, umi=umi, labels=y,
        **{'score__'+k: v for k, v in complete.items()},
        **{'processed_'+mode+'__'+name: x for name, zz in processed.items() for mode, x in zz.items()},
        **{'valid__'+k: v for k, v in valids.items()},
        **{mode+'__'+name: p for mode, pp in default_preds.items() for name, p in pp.items()})
    if boot_by_mode:
        np.savez_compressed(out/f'{section}_bootstrap_f1.npz',
                            **{mode+'__'+name: x for mode, bb in boot_by_mode.items() for name, x in bb.items()})
    ct.dump(out/f'{section}_manifest.json', dict(section=section, inputs=consumed,
        post_freeze=True, hyperparameters_gt_fit=False, label_kind=label_kind,
        original_inference_replays_verified=len(swaps), original_inference_replays_rerun=False,
        processing_label_context_replays=len(replays), all_processing_replays_passed=True,
        grid_size=81, grid_for_all_iterations=True, nulls=list(NULLS),
        grid_mask_row_order=f'{section}_detector_grid.csv', grid_mask_bit_order='numpy.packbits big',
        scope='post_hoc_monotone_null_correction'))
    return dict(section=section, grid=grid, boot=boot_by_mode)


def summarize(results, out):
    grid = pd.concat([x['grid'] for x in results], ignore_index=True)
    default = grid[(grid.seed == 90) & (grid.grow == 80) & (grid.min_size == 5) & (grid['split'] == .4)]
    default.to_csv(out/'headline.csv', index=False)
    keys = ['section', 'mode', 'seed', 'grow', 'min_size', 'split']
    contrasts = []
    for ref in NULLS:
        rr = grid[grid.model.eq(ref)][keys+['f1']].rename(columns={'f1': 'reference_f1'})
        contrast = grid.merge(rr, on=keys, validate='many_to_one')
        contrast['reference'] = ref
        contrast['delta_f1'] = contrast.f1-contrast.reference_f1
        contrasts.append(contrast)
    delta = pd.concat(contrasts, ignore_index=True)
    delta.to_csv(out/'grid_contrasts.csv', index=False)
    sensitivity = []
    for (section, mode, name, ref), df in delta.groupby(['section', 'mode', 'model', 'reference']):
        d = df.delta_f1.dropna()
        sensitivity.append(dict(section=section, mode=mode, model=name, reference=ref,
            n_grid=len(df), n_evaluable_grid=len(d), min_delta=d.min(), median_delta=d.median(),
            max_delta=d.max(), fraction_positive=float((d > 0).mean()) if len(d) else np.nan))
    pd.DataFrame(sensitivity).to_csv(out/'grid_sensitivity.csv', index=False)
    by = {x['section']: x for x in results}
    contexts = ['P6_rep1', 'P4_rep1', 'DCIS1']
    interactions = []
    if set(contexts) <= by.keys():
        for mode in MODES:
            shared = set.intersection(*(set(by[s]['boot'][mode]) for s in contexts))
            for name in sorted(shared):
                for ref in NULLS:
                    effects, actual = [], []
                    for s in contexts:
                        bb = by[s]['boot'][mode]
                        effects.append(bb[name]-bb[ref])
                        dd = default[default.section.eq(s) & default['mode'].eq(mode)].set_index('model')
                        actual.append(dd.loc[name, 'f1']-dd.loc[ref, 'f1'])
                    rng = np.random.default_rng(legacy.SEED+9)
                    dist = effects[0]-(effects[1][rng.permutation(2000)]+effects[2][rng.permutation(2000)])/2
                    interactions.append(dict(endpoint=mode+'_F1', model=name, reference=ref,
                        interaction=actual[0]-(actual[1]+actual[2])/2,
                        ci_low=float(np.quantile(dist, .025)), ci_high=float(np.quantile(dist, .975)),
                        two_sided_bootstrap_p=min(1., 2*min((np.sum(dist <= 0)+1)/2001, (np.sum(dist >= 0)+1)/2001)),
                        scope='section_conditional_P4_copyKAT_silver', multiple_testing_adjusted=False))
    pd.DataFrame(interactions).to_csv(out/'context_interaction.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default='registered_v1')
    ap.add_argument('--source-evaluation-id', default=None)
    ap.add_argument('--evaluation-id', default='evaluation_v2_monotone')
    ap.add_argument('--sections', nargs='+', default=['P6_rep1', 'P6_rep2', 'P4_rep1', 'DCIS1'])
    args = ap.parse_args()
    source_id = args.source_evaluation_id or (args.evaluation_id+'_source_cache')
    for value in [args.run_id, source_id, args.evaluation_id, *args.sections]:
        if not re.fullmatch('[A-Za-z0-9_-]+', value):
            raise ValueError('IDs must be safe single directory names')
    source, out = ct.OUT/source_id, ct.OUT/args.evaluation_id
    if out.exists():
        raise FileExistsError(out)
    if not source.exists():
        subprocess.run([sys.executable, '-m', 'scripts.spatial_viewer_coupling.evaluate',
            '--run-id', args.run_id, '--evaluation-id', source_id, '--sections', *args.sections], check=True)
    em = verify_cache(source, args.run_id)
    if set(args.sections)-set(em['sections']):
        raise ValueError('requested section missing from source cache')
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    inputs = [ct.entry(source/'evaluation_manifest.json', 'sealed_legacy_score_cache'),
              ct.entry(ct.CODE/'MONOTONE_NULL_PROTOCOL.md', 'post_hoc_correction_protocol')]
    sources = [ct.entry(Path(p), 'code') for p in [__file__, controls.__file__, legacy.__file__, audit.__file__, model.__file__, ct.__file__]]
    results = [evaluate_section(ct.OUT/args.run_id/section, source, out) for section in args.sections]
    summarize(results, out)
    audit.check_records(inputs+sources)
    ct.dump(out/'evaluation_manifest.json', dict(schema='monotone_coverage_evaluation_v2',
        run_id=args.run_id, source_evaluation_id=source_id, sections=args.sections, inputs=inputs,
        source_code=sources, runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        nulls=list(NULLS), post_hoc_correction=True, parameters_fit_with_gt=False,
        all_processing_label_swap_tests_passed=True, original_inference_swaps_verified=True,
        elapsed_seconds=time.monotonic()-started,
        unchanged_supporting_results=dict(path=str(source.resolve()),
            files=['replicate_jaccard.csv', 'patient_union_wes.csv', '*_intrinsic_qc.csv', '*_burden_umi.csv', '*_label_swap.csv']),
        files={p.name: ct.sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    (out/'COMPLETE.sha256').write_text(ct.sha(out/'evaluation_manifest.json')+'\n')
    print(f'completed {out}', flush=True)


if __name__ == '__main__':
    main()
