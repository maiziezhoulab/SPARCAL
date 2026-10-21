import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "2_beagle_filtering"
    / "beagle_v2_3_candidate_2026_09_13"
    / "contracts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beagle_reference_ledger_contract", MODULE_PATH
)
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)

FREEZER_PATH = MODULE_PATH.with_name("freeze_reference_ledger.py")
FREEZER_SPEC = importlib.util.spec_from_file_location(
    "beagle_reference_ledger_freezer", FREEZER_PATH
)
FREEZER = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = FREEZER
FREEZER_SPEC.loader.exec_module(FREEZER)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceLedgerContractTests(unittest.TestCase):
    def make_ledger(self, root: Path):
        references = root / "references"
        references.mkdir()
        paths = []
        for index in range(88):
            path = references / f"artifact_{index}.dat"
            path.write_text(f"{index}\n")
            paths.append(path)
        ledger = root / "ledger"
        ledger.mkdir()
        (ledger / "REFERENCE_PATHS.txt").write_text(
            "".join(f"{path.resolve()}\n" for path in paths)
        )
        (ledger / "REFERENCE_SHA256SUMS.txt").write_text(
            "".join(
                f"{digest(path)}  {path.resolve()}\n" for path in paths
            )
        )
        (ledger / "FILE_METADATA.tsv").write_text(
            "path\tsize_bytes\tmtime_epoch\n"
            + "".join(
                f"{path.resolve()}\t{path.stat().st_size}"
                f"\t{int(path.stat().st_mtime)}\n"
                for path in paths
            )
        )
        (ledger / "MANIFEST.txt").write_text(
            "schema=sparcal_beagle_reference_ledger_v1\n"
            "production_inputs_include_wes=false\n"
            "artifact_count=88\n"
        )
        (ledger / "CODE_SHA256SUMS.txt").write_text("synthetic code hash\n")
        local_outputs = (
            "CODE_SHA256SUMS.txt",
            "FILE_METADATA.tsv",
            "MANIFEST.txt",
            "REFERENCE_PATHS.txt",
            "REFERENCE_SHA256SUMS.txt",
        )
        (ledger / "OUTPUT_SHA256SUMS.txt").write_text(
            "".join(
                f"{digest(ledger / name)}  {(ledger / name).resolve()}\n"
                for name in local_outputs
            )
        )
        (ledger / "RUN_COMPLETE").write_text("complete\n")
        return ledger, paths

    def test_complete_ledger_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, paths = self.make_ledger(Path(tmp))
            values = CONTRACTS.verify_reference_ledger(ledger, require_bound_completion=False)
            self.assertEqual(len(values), 88)
            self.assertEqual(values[str(paths[0].resolve())], digest(paths[0]))

    def test_freezer_creates_self_bound_production_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.make_ledger(root)
            output = root / "frozen"
            completion = FREEZER.freeze(source, output)
            self.assertEqual(completion, output / "RUN_COMPLETE.json")
            self.assertEqual(
                len(CONTRACTS.verify_reference_ledger(output)), 88
            )
            with (output / "REFERENCE_SHA256SUMS.txt").open("a") as handle:
                handle.write("mutated\n")
            with self.assertRaisesRegex(
                ValueError, "artifact checksum mismatch"
            ):
                CONTRACTS.verify_reference_ledger(output)

    def test_unbound_ledger_is_rejected_in_production_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, _ = self.make_ledger(Path(tmp))
            with self.assertRaisesRegex(
                ValueError, "lacks bound RUN_COMPLETE.json"
            ):
                CONTRACTS.verify_reference_ledger(ledger)

    def test_changed_reference_is_rejected_without_rehashing_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, paths = self.make_ledger(Path(tmp))
            paths[0].write_text("content and size changed\n")
            with self.assertRaisesRegex(ValueError, "changed since hashing"):
                CONTRACTS.verify_reference_ledger(ledger, require_bound_completion=False)

    def test_mutated_checksum_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, _ = self.make_ledger(Path(tmp))
            with (ledger / "REFERENCE_SHA256SUMS.txt").open("a") as handle:
                handle.write("bad\n")
            with self.assertRaisesRegex(
                ValueError, "output checksum mismatch"
            ):
                CONTRACTS.verify_reference_ledger(ledger, require_bound_completion=False)


if __name__ == "__main__":
    unittest.main()
