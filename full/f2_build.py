"""Phase 0b F2 driver (CPU): build F-B, its GGUF pair, the drafter heads and mimo_FA; copy them into
`data/models/`; write `phase0b/f2_sha256.txt` (checked by every Phase 0b notebook before any probe).

    D40_WORK=/tmp/d40 python f2_build.py --llamacpp /path/to/llama.cpp   # a checkout at FR12's commit, for gguf-py

Run in a fresh D40_WORK. Every build is seeded; the fixtures are committed, so the notebooks never rebuild them.
"""
import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "models"
NEW = ["fb_target", "fb_draft", "fb_target.gguf", "fb_draft.gguf", "eagle3_FA_tgt", "eagle3_FA_flat", "eagle3_FB_tgt",
       "eagle3_FB_flat", "mimo_FA"]
COMMITTED = ["target", "draft", "target.gguf", "draft.gguf"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", required=True)
    a = ap.parse_args()
    work = pathlib.Path(os.environ["D40_WORK"])
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    for n in COMMITTED:
        src = DATA / n
        (shutil.copytree if src.is_dir() else shutil.copy)(src, models / n)
    py = sys.executable
    env = dict(os.environ)
    subprocess.run([py, str(HERE / "fb_fixture.py")], check=True, env=env)
    subprocess.run([py, "vllm_export.py"], cwd=HERE.parent / "code", check=True, env=env)
    subprocess.run([py, str(HERE / "heads.py")], check=True, env=env)
    gg = (f"import sys; sys.path.insert(0, {str(HERE.parent / 'code')!r}); sys.path.insert(0, {str(HERE)!r}); "
          f"sys.path.insert(0, {a.llamacpp + '/gguf-py'!r}); import gguf_export as G, fb_fixture as fb; "
          "[print(G.export(n, fb.config(), fb.V, fb.EOS)) for n in fb.NAMES]")
    subprocess.run([py, "-c", gg], check=True, env=env)
    for n in NEW:
        src, dst = models / n, DATA / n
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)
    lines = []
    for n in COMMITTED + NEW:
        p = DATA / n
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file():
                lines.append(f"{sha(f)}  ../data/models/{f.relative_to(DATA)}")
    out = HERE / "phase0b" / "f2_sha256.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
