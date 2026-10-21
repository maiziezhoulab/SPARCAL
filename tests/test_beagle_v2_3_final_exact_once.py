import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = (
    REPO
    / "clustering_benchmark"
    / "beagle_v2_3_final_strict_regression_2026_09_13"
)
sys.path.insert(0, str(TOOLS))
import exact_once_state as state  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def final_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    candidate = project / "data" / "experiments" / "final_materialization"
    candidate.mkdir(parents=True)
    write_json(
        candidate / "MANIFEST.json",
        {"base_config": "P4", "representation": "single_seed"},
    )
    write_json(candidate / "RUN_COMPLETE.json", {"status": "complete"})
    (candidate / "OUTPUT_SHA256SUMS.txt").write_text("frozen output ledger\n")

    def verify_materialization(root: Path, checked_project: Path):
        assert Path(root).resolve() == candidate.resolve()
        assert Path(checked_project).resolve() == project.resolve()
        return {"root": candidate.resolve()}

    monkeypatch.setattr(
        state.PROVENANCE, "verify_materialization", verify_materialization
    )
    return project, candidate


def claim_and_prepare(project: Path, candidate: Path):
    claimed = state.claim(project=project, candidate_root=candidate)
    run_root = Path(claimed["run_root"])
    run_root.mkdir(parents=True)
    write_json(
        run_root / "STRICT_INPUT_MANIFEST.json",
        {
            "status": "prepared",
            "candidate_semantic_contract": {
                "candidate_root": str(candidate.resolve()),
                "materializer_manifest_sha256": claimed["manifest"]["sha256"],
                "materializer_completion_sha256": claimed["completion"]["sha256"],
            },
            "mandatory_execution_even_if_exactly_equal": True,
        },
    )
    (run_root / "PREP_COMPLETE").write_text("prepared\n")
    state.record_prepared(project=project, run_root=run_root)
    return claimed, run_root


def valid_section(section: str):
    return {
        "status": "all_checks_passed",
        "validated_sections": [section],
        "mandatory_execution_even_if_exactly_equal": True,
    }


def refresh_assessment_ledger(assessment_root: Path) -> None:
    outputs = [
        assessment_root / name
        for name in sorted(state.ASSESSMENT_OUTPUT_NAMES)
    ]
    (assessment_root / "OUTPUT_SHA256SUMS").write_text("".join(
        f"{state.sha256(path)}  {path.name}\n" for path in outputs
    ))


def complete_all_sections(project: Path, run_root: Path) -> None:
    ordered = [state.SMOKE_SECTION] + [
        section for section in state.SECTIONS
        if section != state.SMOKE_SECTION
    ]
    for section in ordered:
        state.start_section(
            project=project, run_root=run_root, section=section
        )
        marker = run_root / section / "CLUSTERING_COMPLETE"
        marker.parent.mkdir(parents=True)
        marker.write_text("complete\n")
        state.complete_section(
            project=project,
            run_root=run_root,
            section=section,
            validator=lambda _root, sections: valid_section(sections[0]),
        )


def write_assessment(assessment_root: Path, *, accepted: bool = True) -> None:
    assessment_root.mkdir(parents=True)
    write_json(
        assessment_root / "NONINFERIORITY_RESULT.json",
        {
            "accept_replacement": accepted,
            "mandatory_execution_even_if_inputs_exactly_equal": True,
        },
    )
    write_json(
        assessment_root / "STRICT_RUN_VALIDATION.json",
        {
            "status": "all_checks_passed",
            "validated_sections": list(state.SECTIONS),
            "mandatory_execution_even_if_exactly_equal": True,
        },
    )
    for name in (
        "clustering_metrics_long.tsv",
        "donor_paired_deltas.tsv",
        "method_success_counts.tsv",
        "section_paired_deltas.tsv",
        "seed_paired_deltas.tsv",
    ):
        (assessment_root / name).write_text(f"{name}\n")
    refresh_assessment_ledger(assessment_root)
    (assessment_root / "ASSESSMENT_COMPLETE").write_text("complete\n")
    decision = "ACCEPTED" if accepted else "REJECTED"
    (assessment_root / decision).write_text("terminal\n")


def test_claim_is_canonical_atomic_and_concurrent(final_materialization):
    project, candidate = final_materialization
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                state.claim, project=project, candidate_root=candidate
            )
            for _ in range(2)
        ]
    successes = [future.result() for future in futures if not future.exception()]
    failures = [future.exception() for future in futures if future.exception()]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    claimed = successes[0]
    key = claimed["claim_key"]
    registry = project / "data" / "experiments" / state.REGISTRY_BASENAME
    assert Path(claimed["run_root"]) == registry / "runs" / key
    assert Path(claimed["assessment_root"]) == registry / "assessments" / key
    assert claimed["mandatory_execution_even_if_exactly_equal"] is True


def test_alternate_root_is_rejected_before_claim(final_materialization):
    project, candidate = final_materialization
    basis = state.materialization_basis(candidate)
    canonical = state.canonical_paths(project, basis)
    with pytest.raises(ValueError, match="alternate run root"):
        state.claim(
            project=project,
            candidate_root=candidate,
            requested_run_root=project / "data" / "experiments" / "alternate",
        )
    assert not Path(canonical["claim_root"]).exists()


def test_identity_drift_blocks_execution(final_materialization):
    project, candidate = final_materialization
    claimed, run_root = claim_and_prepare(project, candidate)
    (candidate / "OUTPUT_SHA256SUMS.txt").write_text("drifted ledger\n")
    with pytest.raises(ValueError, match="identity drift"):
        state.start_section(
            project=project, run_root=run_root, section=state.SECTIONS[0]
        )
    assert claimed["mandatory_execution_even_if_exactly_equal"] is True


def test_full_lifecycle_requires_one_explicit_terminal_acceptance(
    final_materialization,
):
    project, candidate = final_materialization
    claimed, run_root = claim_and_prepare(project, candidate)
    section = state.SMOKE_SECTION
    state.start_section(project=project, run_root=run_root, section=section)
    with pytest.raises(FileExistsError, match="already exists"):
        state.start_section(project=project, run_root=run_root, section=section)
    marker = run_root / section / "CLUSTERING_COMPLETE"
    marker.parent.mkdir(parents=True)
    marker.write_text("complete\n")
    state.complete_section(
        project=project,
        run_root=run_root,
        section=section,
        validator=lambda _root, sections: valid_section(sections[0]),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        state.complete_section(
            project=project,
            run_root=run_root,
            section=section,
            validator=lambda _root, sections: valid_section(sections[0]),
        )
    for remaining in state.SECTIONS:
        if remaining == section:
            continue
        state.start_section(
            project=project, run_root=run_root, section=remaining
        )
        remaining_marker = run_root / remaining / "CLUSTERING_COMPLETE"
        remaining_marker.parent.mkdir(parents=True)
        remaining_marker.write_text("complete\n")
        state.complete_section(
            project=project,
            run_root=run_root,
            section=remaining,
            validator=lambda _root, sections: valid_section(sections[0]),
        )
    assessment_root = Path(claimed["assessment_root"])
    with pytest.raises(ValueError, match="alternate assessment root"):
        state.begin_assessment(
            project=project,
            run_root=run_root,
            assessment_root=assessment_root.parent / "alternate",
        )
    state.begin_assessment(
        project=project,
        run_root=run_root,
        assessment_root=assessment_root,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        state.begin_assessment(
            project=project,
            run_root=run_root,
            assessment_root=assessment_root,
        )
    write_assessment(assessment_root)
    with pytest.raises(FileNotFoundError, match="no explicit terminal"):
        state.verify_acceptance(project=project, run_root=run_root)
    result_path = assessment_root / "NONINFERIORITY_RESULT.json"
    result_payload = json.loads(result_path.read_text())
    result_payload["mandatory_execution_even_if_inputs_exactly_equal"] = False
    write_json(result_path, result_payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        state.complete_assessment(
            project=project,
            run_root=run_root,
            assessment_root=assessment_root,
            decision="accepted",
        )
    refresh_assessment_ledger(assessment_root)
    with pytest.raises(ValueError, match="strict contract"):
        state.complete_assessment(
            project=project,
            run_root=run_root,
            assessment_root=assessment_root,
            decision="accepted",
        )
    result_payload["mandatory_execution_even_if_inputs_exactly_equal"] = True
    write_json(result_path, result_payload)
    refresh_assessment_ledger(assessment_root)
    terminal = state.complete_assessment(
        project=project,
        run_root=run_root,
        assessment_root=assessment_root,
        decision="accepted",
    )
    assert terminal["decision"] == "accepted"
    assert state.verify_acceptance(
        project=project, run_root=run_root
    ) == terminal
    table = assessment_root / "clustering_metrics_long.tsv"
    table.write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        state.verify_acceptance(project=project, run_root=run_root)
    table.write_text("clustering_metrics_long.tsv\n")
    with pytest.raises(FileExistsError, match="already exists"):
        state.complete_assessment(
            project=project,
            run_root=run_root,
            assessment_root=assessment_root,
            decision="accepted",
        )


def test_wrappers_enforce_every_exact_once_phase():
    prepare = (TOOLS / "run_prepare.sh").read_text()
    prepare_python = (TOOLS / "prepare_strict_inputs.py").read_text()
    clustering = (TOOLS / "run_clustering.slurm").read_text()
    assessment = (TOOLS / "run_assessment.slurm").read_text()
    for wrapper in (prepare, clustering, assessment):
        assert wrapper.startswith("#!/bin/bash\n")
    assert "#SBATCH --array=1\n" in clustering
    assert "claim --project" in prepare
    assert "EXACT_ONCE.require_claim" in prepare_python
    assert "EXACT_ONCE.record_prepared" in prepare_python
    assert '"exact_once_state.py"' in prepare_python
    assert "start-section" in clustering and "complete-section" in clustering
    assert clustering.index("start-section") < clustering.index('python "$RUNNER"')
    assert "begin-assessment" in assessment
    assert "complete-assessment" in assessment
    assert "verify-acceptance" in assessment
    assert assessment.index("begin-assessment") < assessment.index("assess_strict_noninferiority.py")
    assert assessment.index("complete-assessment") > assessment.index("finalize_strict_assessment.py")
    assert "sbatch" not in prepare


def test_non_smoke_sections_cannot_start_before_smoke(final_materialization):
    project, candidate = final_materialization
    _, run_root = claim_and_prepare(project, candidate)
    non_smoke = next(
        section for section in state.SECTIONS
        if section != state.SMOKE_SECTION
    )
    with pytest.raises(FileNotFoundError, match="smoke section"):
        state.start_section(
            project=project, run_root=run_root, section=non_smoke
        )
    state.start_section(
        project=project, run_root=run_root, section=state.SMOKE_SECTION
    )
    marker = run_root / state.SMOKE_SECTION / "CLUSTERING_COMPLETE"
    marker.parent.mkdir(parents=True)
    marker.write_text("complete\n")
    state.complete_section(
        project=project,
        run_root=run_root,
        section=state.SMOKE_SECTION,
        validator=lambda _root, sections: valid_section(sections[0]),
    )
    state.start_section(
        project=project, run_root=run_root, section=non_smoke
    )
