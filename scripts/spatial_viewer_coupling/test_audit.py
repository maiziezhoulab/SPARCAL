"""Cross-bundle rejection and known-GT input exclusion."""
import json,tempfile,unittest
from pathlib import Path
from . import audit,contracts as ct

class AuditTests(unittest.TestCase):
    def test_known_gt_rejected_even_if_declared_input(self):
        with tempfile.TemporaryDirectory(dir=ct.OUT) as root:
            p=Path(root)/'tumor_groups.csv';p.write_text('barcode,label\na,1\n')
            with self.assertRaises(ValueError):audit.check_records([ct.entry(p,'numeric_evidence')])
    def test_nested_ledger_hash_checked(self):
        with tempfile.TemporaryDirectory(dir=ct.OUT) as root:
            p=Path(root)/'counts';p.write_text('12');rec=ct.entry(p,'AD')
            ledger=Path(root)/'ledger.json';ct.dump(ledger,[rec]);chain=ct.entry(ledger,'all_inputs_path_sha256');p.write_text('13')
            with self.assertRaises(ValueError):audit.check_records([chain],deep=True)

if __name__=='__main__':unittest.main()
