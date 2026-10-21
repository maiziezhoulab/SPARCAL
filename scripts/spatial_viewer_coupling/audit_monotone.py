"""Independent payload/metric audit of the monotone-null correction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from . import contracts as ct
from .evaluate_monotone import check_evaluation_records, verify_cache
from . import monotone_controls as controls


def main():
    root = ct.OUT
    ev = root/'evaluation_v2_monotone'
    manifest = json.loads((ev/'evaluation_manifest.json').read_text())
    assert ct.sha(ev/'evaluation_manifest.json') == (ev/'COMPLETE.sha256').read_text().strip()
    for name, h in manifest['files'].items():
        assert ct.sha(ev/name) == h, name
    check_evaluation_records(manifest['inputs']+manifest['source_code'])
    old = verify_cache(root/manifest['source_evaluation_id'], manifest['run_id'])
    results = {}
    all_contrasts = pd.read_csv(ev/'grid_contrasts.csv')
    assert set(all_contrasts.reference) == set(controls.NULLS)
    for section in manifest['sections']:
        frame = pd.read_csv(ev/f'{section}_detector_grid.csv')
        assert (frame.groupby(['model', 'mode']).size() == 81).all()
        assert set(controls.NULLS) <= set(frame.model)
        z = ct.checked_npz(ev/f'{section}_maps.npz')
        packed = ct.checked_npz(ev/f'{section}_grid_masks.npz')
        n = len(z['barcode'])
        assert int(packed['n_spots']) == n
        assert np.array_equal(z['barcode'], packed['barcode'])
        masks = np.unpackbits(packed['packed_masks'], axis=1)[:, :n].astype(bool)
        assert len(masks) == len(frame)
        keep = np.isfinite(z['labels'])
        y = z['labels'][keep] > 0
        p = masks[:, keep]
        tp = p[:, y].sum(1)
        fp = p[:, ~y].sum(1)
        fn = int(y.sum())-tp
        for column, value in [('tp', tp), ('fp', fp), ('fn', fn), ('predicted_n', masks.sum(1))]:
            np.testing.assert_array_equal(frame[column], value)
        if keep.any():
            f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(tp), float), where=(2*tp+fp+fn) > 0)
            np.testing.assert_allclose(frame.f1, f1, atol=1e-15)
        else:
            assert frame.f1.isna().all()
        default = frame[(frame.seed == 90) & (frame.grow == 80) & (frame.min_size == 5) & (frame['split'] == .4)]
        for i, row in default.iterrows():
            np.testing.assert_array_equal(z[f'{row["mode"]}__{row.model}'], masks[i])
        for name in controls.NULLS:
            np.testing.assert_array_equal(rankdata(z['score__'+name]), rankdata(z['score__smooth_coverage']))
        if section == 'P6_rep1':
            np.testing.assert_array_equal(rankdata(z['score__coverage_Coff_k0']), rankdata(z['score__smooth_coverage']))
        sm = json.loads((ev/f'{section}_manifest.json').read_text())
        check_evaluation_records(sm['inputs'])
        swaps = pd.read_csv(ev/f'{section}_processing_label_swap.csv')
        assert swaps.pass_exact.all() and swaps.io_guard_pass.all()
        if keep.any():
            paired = pd.read_csv(ev/f'{section}_paired_spatial.csv')
            assert set(paired.reference) == set(controls.NULLS)
            assert len(paired) == len(default)*2
            boot = ct.checked_npz(ev/f'{section}_bootstrap_f1.npz')
            actual = default.set_index(['mode', 'model']).f1
            for row in paired.itertuples():
                delta = actual.loc[row.mode, row.model]-actual.loc[row.mode, row.reference]
                assert np.isclose(delta, row.delta_f1, atol=1e-15)
                d = boot[row.mode+'__'+row.model]-boot[row.mode+'__'+row.reference]
                np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(d, [.025, .975]), atol=1e-15)
        results[section] = dict(n_models=frame.model.nunique(), grid_rows=len(frame), metric_recount_pass=True,
            original_replays_verified=sm['original_inference_replays_verified'], processing_replays=len(swaps))
    report = json.loads((root/'report_manifest.json').read_text())
    for name, h in report['files'].items():
        assert ct.sha(root/name) == h, name
    check_evaluation_records(report['inputs'])
    before = json.loads((root/'monotone_revision_preflight.json').read_text())
    protected = {p: ct.sha(p) == h for p, h in before['protected'].items()}
    original_source = {p: ct.sha(p) == h for p, h in before['source_before'].items()}
    allowed_changes = {'README.md', 'evaluate_registered.slurm'}
    assert all(ok or Path(p).name in allowed_changes for p, ok in original_source.items())
    archive = {p: ct.sha(p) == h for p, h in before['archive'].items()}
    assert all(archive.values())
    assert ct.sha(ct.CODE/'MONOTONE_NULL_PROTOCOL.md') == before['protocol_sha256']
    payload = dict(sealed_evaluation_verified=True, original_evaluation_verified=True,
        report_verified=True, sections=results, protected_files_unchanged=protected,
        all_protected_unchanged=all(protected.values()), source_before_unchanged=original_source,
        old_report_archive_unchanged=archive, protocol_unchanged=True,
        inputs=[ct.entry(ev/'evaluation_manifest.json', 'corrected_evaluation'),
                ct.entry(root/'report_manifest.json', 'corrected_report'),
                ct.entry(Path(__file__), 'audit_code')])
    ct.dump(root/'final_audit_monotone.json', payload)
    print(json.dumps(dict(total_grid_cells=sum(r['grid_rows'] for r in results.values()),
        original_replays=sum(r['original_replays_verified'] for r in results.values()),
        new_processing_replays=sum(r['processing_replays'] for r in results.values()),
        protected_unchanged=payload['all_protected_unchanged']), indent=2))


if __name__ == '__main__':
    main()
