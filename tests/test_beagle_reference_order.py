import gzip
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
path=PROJECT/"clustering_benchmark/beagle_default_dlpfc_2026_09_19/align_reference_order.py"
spec=importlib.util.spec_from_file_location("order_adapter",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def row(pos,ref,alt):
    return f"chr1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT:PL\t0/1:30,0,30\n"

class OrderTests(unittest.TestCase):
    def test_reference_order_preserves_records(self):
        rows=[row(100,"C","G"),row(100,"C","T")]
        order=[("chr1",100,"C","T"),("chr1",100,"C","G")]
        self.assertEqual(module.reorder(rows,order),rows[::-1])

    def test_unmatched_alleles_keep_relative_order(self):
        rows=[row(100,"C","A"),row(100,"C","T"),row(100,"C","G")]
        self.assertEqual(module.reorder(rows,[("chr1",100,"C","T")]),[rows[1],rows[0],rows[2]])

    def test_bad_target_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"raw.vcf.gz"
            for rows,expected in [([row(200,"A","T"),row(100,"A","G")],"sorted"),
                                  ([row(100,"A","G"),row(100,"A","G")],"duplicate")]:
                with gzip.open(p,"wt") as f: f.writelines(rows)
                with self.assertRaisesRegex(ValueError,expected): list(module.groups(p))

    def test_actual_beagle_recovers_skipped_marker_after_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            header='##fileformat=VCFv4.2\n##contig=<ID=chr1,length=100000>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihood">\n'
            markers=[(100,"C","T"),(100,"C","G"),(200,"A","T"),(300,"A","C")]
            ref=root/"reference.vcf"
            ref.write_text(header+'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tR1\tR2\tR3\tR4\n'+''.join(
                f'chr1\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\tGT\t0|0\t0|1\t1|0\t1|1\n' for p,r,a in markers))
            refgz=root/"reference.vcf.gz"
            with refgz.open("wb") as f:
                subprocess.run([str(PROJECT/"apps/bgzip"),"-c",str(ref)],stdout=f,check=True)
            subprocess.run([str(PROJECT/"apps/tabix"),"-p","vcf",str(refgz)],check=True)
            source=root/"raw.vcf.gz"
            with gzip.open(source,"wt") as f:
                f.write(header+'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT1\n')
                f.writelines(row(*markers[i]) for i in [1,0,2,3])
            aligned=root/"adapted/target.vcf.gz"
            result=module.align(source,aligned,{"chr1":refgz},PROJECT/"apps/bcftools",PROJECT/"apps/bgzip",PROJECT/"apps/tabix")
            self.assertEqual(result["reordered_positions"],1)
            self.assertEqual(result["records"],4)
            self.assertTrue(result["per_position_record_multiset_exact"])
            counts=[]
            for name,target in [("old",source),("new",aligned)]:
                command=[str(PROJECT/"apps/src/jdk-11.0.2/bin/java"),"-Xmx512m","-jar",str(PROJECT/"apps/beagle.27Jul16.86a.jar"),
                    "gl="+str(target),"ref="+str(refgz),"out="+str(root/name),"impute=false","modelscale=2","maxlr=5000",
                    "nthreads=1","gprobs=true","niterations=0","seed=20260915"]
                subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
                with gzip.open(root/(name+".vcf.gz"),"rt") as f:
                    counts.append(sum(not line.startswith("#") for line in f))
            self.assertEqual(counts,[3,4])

if __name__=="__main__": unittest.main()
