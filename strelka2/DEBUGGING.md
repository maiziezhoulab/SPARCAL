# Strelka2 Germline DLPFC — Debugging Log

## 2026-05-24 — Jobs stagnated, two bugs found and fixed

### Background

Array job `sbatch --array=0-11 run_slurm/strelka2_germline_dlpfc.sh` was submitted to
run Strelka2 germline calling on all 12 DLPFC sections (151507–151676) using merged
per-section BAMs (~17 GB each). Jobs appeared RUNNING in squeue for 15+ hours but
produced zero output.

---

### Bug 1: SMTP hang at startup (fixed)

**Symptom:** Jobs ran for 15+ hours, workspace directory stayed completely empty,
no pyflow output anywhere.

**Root cause:** `runWorkflow.py` calls `isLocalSmtp()` during argument parsing
(before pyflow initializes), which does:

```python
smtplib.SMTP('localhost')   # no timeout
```

On compute nodes (cn1287, cn1814, etc.), port 25 accepts the TCP connection but never
sends an SMTP greeting banner. Python 2's `smtplib.SMTP()` with no timeout blocks
forever. The login node (`maizie.vampire`) doesn't have this problem, which is why
the older per-barcode strelka2 runs (Jul 2025) all succeeded.

**Fix:** Added `timeout=5` in three places:

- `strelka-2.9.2.centos6_x86_64/lib/python/makeRunScript.py` line 136
  (template that generates `runWorkflow.py` — fixes all future configure runs)
- `strelka-2.9.2.centos6_x86_64/lib/python/pyflow/pyflow.py` line 406
  (pyflow's own email check during workflow execution)
- All 12 already-generated `runWorkflow.py` files (sections 151507–151676)
  patched in-place with `sed`

**Job history:** Stagnated jobs cancelled (11272152). First fixed resubmission
failed because I (Claude) submitted without the `strelka_py2` conda environment,
so Python 3 was used and the configure step immediately errored. User resubmitted.

---

### Bug 2: Reference FASTA path unresolvable on panfs (fixed via /tmp copy)

**Symptom:** After Bug 1 was fixed, jobs progressed through `getChromDepth` but
then every `GetSequenceErrorCounts` task failed with:

```
COMMAND-LINE ERROR:: can't resolve reference path:
  /data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa
```

Tasks were retried twice each (pyflow retry policy) then the workflow aborted.
All 12 sections failed (jobs 11287046, 11287211, all exit code 1).

**Root cause:** Strelka2's C++ binary (`GetSequenceErrorCounts`, `strelka2`) calls
`boost::filesystem::canonical()` — backed by the OS `realpath()` syscall — to resolve
the reference FASTA path. This fails on compute nodes for panfs **symlink** paths:

```
/data/maiziezhou_lab/...            ← symlink → fails
/panfs/accrepfs.vampire/data/...    ← canonical → works (confirmed by job 11390056)
```

(Note: an earlier claim that the canonical `/panfs/...` form also fails was incorrect;
it was based on job 11287211 which failed for a different reason — see Bug 4.)

The Python configure step uses `os.path.exists()` which follows symlinks fine at the
Python/kernel level, so configure succeeded. The C++ runtime tasks fail because
`boost::filesystem::canonical()` cannot follow the `/data/maiziezhou_lab` symlink on
compute nodes.

**Fix:** Copy the reference to the compute node's local `/dev/shm` (RAM-backed tmpfs,
guaranteed real path) before running the workflow. The configure step is given
`--referenceFasta /dev/shm/strelka2_ref/genome.fa`, which the C++ binary resolves fine.

---

## 2026-05-27 — /tmp copy approach also failing; root cause identified and fixed

### Bug 3: shutil.copy2 silently writes empty/truncated genome.fa to /tmp

**Symptom:** After switching to the `/tmp` copy approach, all 12 sections continued
to fail with the same error (jobs 11287522 and array, all exit code 1):

```
COMMAND-LINE ERROR:: can't resolve reference path: /tmp/strelka2_ref/genome.fa
```

Python logged `"Reference ready at local path: /tmp/strelka2_ref/genome.fa"` before
the workflow started, so the copy appeared to succeed.

**Diagnosis (2026-05-27):**

Three test jobs were submitted to isolate the issue:

- **Job 11389914** (`test_tmp_access.sh`): Ran on cn1287. Confirmed `/tmp` is a real
  local ext4 disk (726 GB, `/dev/mapper/vg.os-lv.tmp`), not a panfs mount. Both
  Python 2 (`strelka_py2` env) and Python 3 (`snv_caller` env) can write to `/tmp`
  and `libc.realpath()` works correctly. The strelka2 C++ binary loads fine.

- **Job 11390056** (`test_strelka2_ref_on_compute.sh`): Ran on cn1287. Copied
  `genome.fa` (3.0 GB) using `cp`, verified size matches source. Then ran
  `GetSequenceErrorCounts --ref /tmp/strelka2_ref/genome.fa` directly — result was
  `"Must specify at least one input alignment file"`, meaning the reference path
  **resolved correctly**. The `/tmp` path itself is not the problem.

**Root cause (two parts):**

1. **`shutil.copy2` silently produces empty/truncated genome.fa.** Python 2's
   `shutil.copy2` copies in 16 KB chunks. If panfs I/O is degraded at copy time,
   reads can return early/empty without raising a Python exception. The destination
   file is created (so `os.path.exists()` returns True) but contains 0 bytes or
   truncated data. The C++ binary then fails when it tries to initialize the
   reference from the empty file.

2. **Skip-if-exists guard reuses the bad file on every retry.** The old copy code:
   ```python
   if os.path.exists(src) and not os.path.exists(dst):
       shutil.copy2(src, dst)  # skipped if dst exists, even if 0 bytes!
   ```
   Once a bad `genome.fa` was written to `/tmp/strelka2_ref/` on a node, every
   subsequent job on that same node skipped the copy and reused the bad file,
   creating a persistent failure loop regardless of how many times the job was
   resubmitted.

**Fix (applied 2026-05-27 in `scripts/run_strelka2_germline.py`):**

- Replace `shutil.copy2` with `subprocess.call(['cp', src, dst])`. `cp` is more
  robust for large panfs files and returns a non-zero exit code on failure.
- Skip the copy only if destination **size matches source** (not just "file exists").
- Verify destination size after copy; abort with a clear error if there is a mismatch.

```python
src_size = os.path.getsize(src)
dst_size = os.path.getsize(dst) if os.path.exists(dst) else -1
if dst_size == src_size:
    log.info("Already present and correct size, skipping: %s", filename)
    continue
rc = subprocess.call(["cp", src, dst])
if rc != 0:
    sys.exit("Failed to copy %s (exit %d)" % (filename, rc))
dst_size = os.path.getsize(dst)
if dst_size != src_size:
    sys.exit("Size mismatch for %s: src=%d dst=%d" % (filename, src_size, dst_size))
```

---

### Current status after Bug 3 fix (2026-05-27)

Jobs 11390244 (task 0, 151507, cn1328) and 11390247 (tasks 1–11) both FAILED — same
"can't resolve reference path: /tmp/strelka2_ref/genome.fa" error. Bug 3 was real
(shutil.copy2 truncation) but not the last bug.

---

## 2026-05-28 — Root cause finally isolated: Bug 4

### Bug 4: Symlink paths in --align-file and --chrom-depth-file trigger "can't resolve reference path"

**Symptom:** Identical error persists after Bug 3 fix. The `/tmp` reference path is
valid and correct size, yet every `GetSequenceErrorCounts` task fails.

**Diagnosis (jobs 11390594, 11390722, 11390922):**

A series of direct-bash tests (no pyflow, same arguments) proved:

- **Without `--align-file`** (minimal args `--ref /tmp/... --max-indel-size 49`):
  binary exits 1 with **"Must specify at least one input alignment file"** — reference
  resolved correctly, binary got past path validation. ✓

- **With `--align-file /data/maiziezhou_lab/...` (symlink path)**:
  binary exits 1 with **"can't resolve reference path: /tmp/strelka2_ref/genome.fa"**
  — the error message always names the reference regardless of which path failed. ✗

- **LD_LIBRARY_PATH has no effect**: tested with full env, unset, and apps/ stripped —
  identical failures. Confirmed by `ldd`: binary only links libz, librt, libpthread,
  libm, libc — nothing from apps/. ✗

**Root cause:**

`boost::filesystem::canonical()` is called on every input path argument. It fails when
given a panfs symlink path (`/data/maiziezhou_lab/... → /panfs/accrepfs.vampire/...`)
but succeeds on the canonical `/panfs/accrepfs.vampire/...` form (confirmed: job 11390056
passed `--ref /panfs/accrepfs.vampire/.../genome.fa` and got "Must specify at least one
input alignment file" — meaning resolution succeeded).

pyflow stores and re-uses whichever paths were given at configure time. Because
`run_strelka2_germline.py` passed `--bam /data/maiziezhou_lab/.../151507_merged.bam`
and `--runDir /data/maiziezhou_lab/.../strelka2`, pyflow recorded symlink paths in
every task command:

```
GetSequenceErrorCounts ... \
  --align-file /data/maiziezhou_lab/.../151507_merged.bam \          # symlink → fails
  --chrom-depth-file /data/maiziezhou_lab/.../workspace/chromDepth.tsv  # symlink → fails
```

The error is always reported as "can't resolve reference path" even when the failing
argument is `--align-file` — a strelka2 error-reporting quirk, not the actual failed path.

**Fix (iteration 1 — partial, applied 2026-05-28):**

Applied `os.path.realpath()` to BAM and runDir. pyflow task commands now use canonical
paths for `--align-file` and `--chrom-depth-file`. Confirmed via pyflow pickle inspection.
Jobs 11391134 (array) still failed — now with `can't resolve reference path: /dev/shm/...`.

**Addendum (2026-05-28): `/dev/shm` is also a symlink on ACCRE compute nodes.**

`/dev/shm` → `/run/shm` on RHEL9. `boost::filesystem::canonical()` cannot follow this
symlink either. The copy-to-/dev/shm approach (Bugs 2+3) was therefore never fully
correct — it only moved the failing argument from `--align-file` to `--ref`.

**Fix (iteration 2 — final, applied 2026-05-28):**

Remove the reference file copy entirely. Use `os.path.realpath(REFERENCE_FA)` for the
reference, giving the canonical `/panfs/accrepfs.vampire/...` path that the binary CAN
resolve. Same approach as the BAM and runDir fix.

```python
reference_fa_real = os.path.realpath(REFERENCE_FA)

merged_bam_real  = os.path.realpath(merged_bam)
strelka_dir_real = os.path.realpath(strelka_dir)

configure_cmd = [
    "python", STRELKA2_BIN,
    "--bam",            merged_bam_real,
    "--referenceFasta", reference_fa_real,   # canonical panfs path — no copy needed
    "--runDir",         strelka_dir_real,
]
if call_regions:
    configure_cmd += ["--callRegions", os.path.realpath(call_regions)]
```

**Summary:** Every path passed to `configureStrelkaGermlineWorkflow.py` must go through
`os.path.realpath()`. pyflow inherits whatever paths configure records.

**Note:** DEBUGGING.md Bug 2 stated that canonical `/panfs/...` paths also fail — this
was incorrect. Testing (job 11390056) confirmed only the symlink form fails; the canonical
form works fine.

---

### Current status (2026-05-28)

Bug 4 fix was a dead end. See Bug 5 below — the root cause is deeper.

---

## 2026-05-28 — Bug 5: boost::filesystem::canonical() fails inside GetSequenceErrorCounts regardless of path form

### Summary of all diagnostic tests (jobs 11391978, 11393830, 11394713)

After applying all path fixes (Bugs 1–4), `GetSequenceErrorCounts` still fails.
A systematic isolation campaign revealed the following:

| Test | ref form | BAM form | all required args? | result |
|------|----------|----------|-------------------|--------|
| 11390056 | `/tmp/` local | none | no | PASSES ("Must specify alignment file") |
| arg_isolation test 0 | `/panfs/...` canonical | none | no | PASSES |
| arg_isolation tests 1–7 | any | any | **no** (missing `--counts-file` etc.) | PASSES with validation error |
| arg_isolation test 8 | `/panfs/...` canonical | `/panfs/...` canonical | **yes** | FAILS |
| arg_isolation test 9 | `/panfs/...` canonical | local `/tmp/` | **yes** | FAILS |
| realpath_tmpref C1 | `/tmp/` local (verified correct size) | `/panfs/...` canonical | **yes** | FAILS |
| realpath_tmpref D1 | `/panfs/...` canonical | `/panfs/...` canonical | **yes** | FAILS |
| realpath_tmpref E1 | `/data/...` symlink | `/panfs/...` canonical | **yes** | FAILS |

**Key pattern:** The binary exits with `"COMMAND-LINE ERROR:: can't resolve reference path: <whatever --ref was>"` if and only if ALL required arguments are supplied (i.e., `--align-file`, `--counts-file`, `--nonempty-site-count-file` are all present). With any required argument missing, the binary exits with a different validation error and never hits this code path.

### C realpath() test (job 11394713, Section A)

A small C program compiled with the system gcc (glibc 2.34, RHEL9) and run on cn1287 called `realpath()` on every relevant path. **All 7 calls succeeded with 0 failures**, including:
- canonical panfs genome.fa, genome.fa.fai, genome.dict
- canonical panfs 151507_merged.bam, chromDepth.tsv
- symlink `/data/maiziezhou_lab/...` genome.fa (correctly resolved to canonical form)
- local `/tmp`

**Conclusion: the system glibc `realpath()` works perfectly on the compute nodes for ALL path forms. The path resolution failure is NOT in glibc.**

### Root cause

The error originates from `boost::filesystem::canonical()` inside the strelka2 CentOS6 binary. This is called (in the `COMMAND-LINE ERROR` path) after all argument validation passes, when the binary actually tries to initialize the reference genome. Old boost builds (pre-1.60) implement `canonical()` with a manual path-traversal algorithm rather than a simple `realpath()` call; the two implementations can behave differently depending on kernel/filesystem configuration.

**The failure is node-level and environment-level, not path-level.** Every path form tested (local `/tmp`, canonical panfs, symlink panfs) produces the same error. The problem is a compatibility break between the CentOS6 strelka2 binary and the current RHEL9 compute node environment, not a path format issue.

### Additional observations

- `GetChromDepth` (same strelka2 2.9.2 package) runs successfully on the same compute nodes — chromDepth.tsv is produced correctly. `GetChromDepth` does not process a reference FASTA. The reference initialization code path is specific to `GetSequenceErrorCounts` and downstream binaries.
- `strace` is blocked on compute nodes: `PTRACE_TRACEME: Operation not permitted`. This prevents syscall-level diagnosis. The block itself may be a SECCOMP policy enforced by the current SLURM/OS configuration.
- Jobs complete in < 1 second when they fail — the binary never reads any file data; it fails during argument processing / library initialization.
- Strelka2 germline on these BAMs previously succeeded (Jul 2025) on `maizie.vampire` (login node). Current runs are on compute nodes (cn1287 etc., RHEL9).

### Likely causes (for ACCRE admin)

1. **SECCOMP policy**: A SECCOMP filter added or tightened in the recent system upgrade blocks one or more syscalls that old boost::filesystem uses (e.g., old variants of `stat`, `getdents`, or thread-creation syscalls). `GetChromDepth` does not trigger these paths; `GetSequenceErrorCounts` does.
2. **CentOS6 / RHEL9 ABI mismatch**: The binary was compiled against glibc 2.12. Running on RHEL9 (glibc 2.34), certain deprecated or versioned glibc symbols behave differently, particularly in the boost path-traversal code.
3. **Filesystem or mount configuration change**: A change in how panfs is mounted on compute nodes (options, namespace, or kernel NFS/panfs driver version) that affects old boost's `is_symlink()` / `exists()` calls differently from glibc's `realpath()`.
4. **Missing capability or resource limit**: A capability (e.g., `CAP_SYS_ADMIN` for certain filesystem operations) or resource limit (memory map size, file descriptor count) that the binary needs but is no longer granted on compute nodes.

### Next steps

- Email ACCRE help desk (see `accre_admin_email.txt`) describing the symptoms and asking for (a) information on what changed in the recent upgrade, and (b) whether SECCOMP filtering is in place and if ptrace can be enabled for diagnostic purposes.
- As a workaround: attempt to run strelka2 on the **login node** (maizie.vampire) directly (not via SLURM) to confirm it still works there. If yes, request an interactive session or exemption.
- If admin confirms SECCOMP: request that the strelka2 binary be added to an allowlist, or rebuild strelka2 from source against RHEL9 (boost 1.75+).

---

## 2026-06-01 — RESOLVED. Root cause was the prebuilt binary itself; fixed by a bioconda rebuild.

### Two corrections to the Bug 5 writeup above

1. **It was NEVER compute-node-specific.** The Bug 5 analysis rested on "strelka worked
   on the login node (Jul 2025)" — but that belief was stale (pre-OS-upgrade). On 2026-06-01
   the OLD `GetSequenceErrorCounts` binary was run directly on the login node (gw01) against
   the bundled 5 KB demo data (short, local, real path) and **it failed there too** with the
   identical `can't resolve reference path: .../demo20.fa`. So SECCOMP / cgroups / panfs-mount
   theories are all moot — the binary is simply broken against the current OS, everywhere.

2. **All the path chasing (Bugs 2–4: panfs symlink, /tmp, /dev/shm, os.path.realpath) was
   attacking a phantom.** The failure reproduces on a 5 KB demo file at a short local path with
   no panfs involvement. Path format was never the issue.

### The fix (bioconda rebuild — see `scripts/install_strelka_conda.sh`)

`conda create -n strelka -c bioconda -c conda-forge strelka=2.9.10=hdfd78af_2 python=2.7`
gives a Nov-2024 rebuild. Even though its ELF profile looks near-identical to the broken
binary (still "GNU/Linux 2.6.18" ABI tag, still static boost, same `GLIBC_2.7` ceiling), the
rebuilt binary **resolves the reference and runs clean**. Confirmed two ways on 2026-06-01:

- Login-node demo: conda `GetSequenceAlleleCounts` (the 2.9.10 analog of the old
  `GetSequenceErrorCounts`) on demo data → exit 0, produced counts files.
- Full workflow on a compute node (cn1330, job **11528358_0**, section 151507): COMPLETED,
  exit 0, 11m14s, ~6.4 GB RSS. `EstimateSeqErrorParams` + `estimateVariantErrorRates` (the
  step that used to die) ran clean; `variants.vcf.gz` = 317,492 records / 102,292 PASS across
  all 24 GRCh38 contigs, integrity OK.

NOTE: a fresh conda strelka ships WITHOUT the Bug 1 SMTP-timeout patch, so
`install_strelka_conda.sh` re-applies it to the env's `pyflow.py` and `makeRunScript.py`.
Without that, runWorkflow.py would hang on compute nodes again.

### Current wiring
- `scripts/run_strelka2_germline.py` resolves the configure script via `$STRELKA_CONFIG` →
  PATH → (legacy fallback). No path-munging needed.
- `run_slurm/strelka2_germline_dlpfc.sh` activates env `strelka` and exports `STRELKA_CONFIG`.
- The old `strelka-2.9.2.centos6_x86_64` tree is abandoned (kept only for reference).
