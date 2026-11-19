"""SHA-256 manifest of every downloaded data file, checked against the publishers' own checksums.

    python data_manifest.py            # hash data/, fetch the publishers' checksum lists, compare

For each downloaded file in data/ this records its SHA-256 and checks it
against what the publisher posts:
  - PhysioNet projects: SHA-256, from the project's SHA256SUMS.txt
    (Challenge 2020 for CPSC 2018, CPSC 2018 extra and Georgia; Challenge
    2021 for Chapman-Shaoxing and Ningbo; PTB-XL; INCART; MIMIC-IV-ECG).
    Local folders are flattened, so files are matched by name within the
    project's folder (Ningbo's local "gN_" prefix removed); a name that
    occurs more than once there can't be matched and is reported.
  - Zenodo (CODE-15%, the CODE test set, Ribeiro's model) and figshare
    (SPH): MD5, from their APIs.
  - Files unpacked from a downloaded archive (CODE-15% .hdf5, the CODE test
    set and model folders, SPH records/, Challenge 2017 training2017/):
    compared with the archive member itself (zip: size and CRC-32; tar:
    SHA-256), so publisher checksum -> archive -> unpacked file.
  - Challenge 2017: PhysioNet posts no checksum list for version 1.0.0, so
    its zip is only hashed.
The project's own files in data/ (feature caches *.pkl/*.npz, the Holter
.npy caches, SELECTED lists, download lists) aren't downloads and are left
out. The Holter databases (AFDB, LTAFDB, SVDB) were streamed by wfdb and
only their .npy caches are on disk, so they can't be checked. The
publishers' own checksum lists (INCART's SHA256SUMS.txt) are left out too.
Local hashes are cached in data/_checksums/local_hashes.pkl by path, size
and modification time, so a rerun only hashes new or changed files.

Writes results/data_sha256sums.txt.gz (sha256sum format: check with
`gunzip -c results/data_sha256sums.txt.gz | shasum -a 256 -c`, from the
repository root), results/data_manifest_summary.csv (per dataset) and
results/data_manifest_problems.csv (every file that didn't verify).
"""

import gzip
import hashlib
import io
import json
import os
import re
import ssl
import tarfile
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

DATA = "data"
CACHE = "data/_checksums"
PN = "https://physionet.org/files/"
DERIVED = re.compile(r"(\.pkl|\.npz|\.npy)$|(^|/)SELECTED|(^|/)urls[^/]*\.cfg$|^_checksums/|(^|/)SHA256SUMS\.txt$")

SOURCES = {
    "cpsc2018": ("physionet", "challenge-2020/1.0.2", "training/cpsc_2018/"),
    "cpsc2018_extra": ("physionet", "challenge-2020/1.0.2", "training/cpsc_2018_extra/"),
    "georgia": ("physionet", "challenge-2020/1.0.2", "training/georgia/"),
    "chapman": ("physionet", "challenge-2021/1.0.3", "training/chapman_shaoxing/"),
    "ningbo": ("physionet", "challenge-2021/1.0.3", "training/ningbo/"),
    "ptbxl": ("physionet-path", "ptb-xl/1.0.3", ""),
    "incart": ("physionet-path", "incartdb/1.0.0", ""),
    "mimic": ("physionet-path", "mimic-iv-ecg/1.0", ""),
    "cinc2017": ("none", "challenge-2017/1.0.0 (no checksum list posted)", ""),
    "code15": ("zenodo", [4916206], ""),
    "ribeiro": ("zenodo", [3625006, 3625017], ""),
    "sph": ("figshare", [17912444, 17912441, 17912507], ""),
}
ARCHIVES = {  # archive (relative to data/) -> folder its members were unpacked into (relative to data/)
    "cinc2017/training2017.zip": "cinc2017",
    "ribeiro/data.zip": "ribeiro",
    "ribeiro/model.zip": "ribeiro",
    "sph/records.tar.gz": "sph",
    **{f"code15/exams_part{k}.zip": "code15" for k in range(18)},
}
HOLTER = ["afdb", "ltafdb", "svdb"]


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        with urllib.request.urlopen(url, context=_ctx(), timeout=300) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + ".part", path)
    return path


def physionet_sums(project):
    path = fetch(f"{PN}{project}/SHA256SUMS.txt", project.replace("/", "_") + "_SHA256SUMS.txt")
    out = {}
    for line in open(path):
        h, _, p = line.rstrip("\n").partition(" ")
        if p:
            out[p.strip()] = h
    return out


def publisher_md5(kind, ids):
    out = {}
    for i in ids:
        if kind == "zenodo":
            r = json.load(open(fetch(f"https://zenodo.org/api/records/{i}", f"zenodo_{i}.json")))
            out.update({f["key"]: f["checksum"].removeprefix("md5:") for f in r["files"]})
        else:
            r = json.load(open(fetch(f"https://api.figshare.com/v2/articles/{i}", f"figshare_{i}.json")))
            out.update({f["name"]: f["computed_md5"] for f in r["files"]})
    return out


def hash_file(path, md5=False, crc=False):
    s, m, c = hashlib.sha256(), hashlib.md5() if md5 else None, 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            s.update(chunk)
            if m:
                m.update(chunk)
            if crc:
                c = zlib.crc32(chunk, c)
    return s.hexdigest(), m.hexdigest() if m else None, c if crc else None


def archive_members():
    """{local path relative to data/: ("zip", size, crc) or ("tar", sha256)} for every member of a local archive."""
    out = {}
    for arch, into in ARCHIVES.items():
        p = os.path.join(DATA, arch)
        if not os.path.exists(p):
            continue
        if arch.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    if not i.is_dir():
                        out[os.path.normpath(os.path.join(into, i.filename))] = ("zip", i.file_size, i.CRC, arch)
        else:
            with tarfile.open(p) as t:
                for m in t:
                    if m.isfile():
                        h = hashlib.sha256()
                        f = t.extractfile(m)
                        while chunk := f.read(1 << 22):
                            h.update(chunk)
                        out[os.path.normpath(os.path.join(into, m.name))] = ("tar", h.hexdigest(), None, arch)
    return out


if __name__ == "__main__":
    files = sorted(os.path.relpath(os.path.join(r, f), DATA) for r, _, fs in os.walk(DATA) for f in fs
                   if not f.startswith(".") and not f.endswith(".part"))
    downloads = [f for f in files if not DERIVED.search(f) and f.split("/")[0] in SOURCES]
    print(f"{len(files)} files in data/; {len(downloads)} downloaded files to hash")
    members = archive_members()
    print(f"{len(members)} archive members listed")
    top_md5 = {f for f in downloads if f.count("/") == 1 and SOURCES[f.split("/")[0]][0] in ("zenodo", "figshare")}

    cache_path = os.path.join(CACHE, "local_hashes.pkl")      # (path, size, mtime) -> hashes, to avoid rehashing
    cache = pd.read_pickle(cache_path) if os.path.exists(cache_path) else {}

    def work(f):
        p, st = os.path.join(DATA, f), os.stat(os.path.join(DATA, f))
        key = (f, st.st_size, st.st_mtime_ns)
        if key not in cache:
            cache[key] = hash_file(p, md5=f in top_md5, crc=members.get(f, ("",))[0] == "zip")
        return f, st.st_size, *cache[key]
    with ThreadPoolExecutor(8) as pool:
        hashed = list(pool.map(work, downloads))
    pd.to_pickle(cache, cache_path)
    print("hashed")

    pn, md5s = {}, {}
    rows = []
    for f, size, sha, md5, crc in hashed:
        ds, rest = f.split("/", 1)
        kind, spec, prefix = SOURCES[ds]
        if f in members:
            m = members[f]
            ok = (m[1] == size and m[2] == crc) if m[0] == "zip" else (m[1] == sha)
            status, ref = ("unpacked: matches archive member" if ok else "unpacked: DIFFERS from archive member"), m[3]
        elif kind in ("physionet", "physionet-path"):
            if (spec, prefix) not in pn:
                sums = physionet_sums(spec)
                by_name = {}
                for p, h in sums.items():
                    if p.startswith(prefix):
                        by_name.setdefault(p.rsplit("/", 1)[-1], []).append((p, h))
                pn[spec, prefix] = (sums, by_name)
            sums, by_name = pn[spec, prefix]
            if kind == "physionet-path":
                hit = [(rest, sums[rest])] if rest in sums else []
            else:
                name = re.sub(r"^g\d+_", "", rest.rsplit("/", 1)[-1]) if ds == "ningbo" else rest.rsplit("/", 1)[-1]
                hit = by_name.get(name, [])
            if len(hit) == 1:
                status, ref = ("publisher SHA-256 match" if hit[0][1] == sha else "publisher SHA-256 MISMATCH"), hit[0][0]
            elif len(hit) > 1:
                status, ref = "name occurs more than once in publisher list", ""
            else:
                status, ref = "not in publisher list", ""
        elif kind in ("zenodo", "figshare"):
            if ds not in md5s:
                md5s[ds] = publisher_md5(kind, spec)
            if rest in md5s[ds]:
                status, ref = ("publisher MD5 match" if md5s[ds][rest] == md5 else "publisher MD5 MISMATCH"), rest
            else:
                status, ref = "not in publisher list", ""
        else:
            status, ref = "no publisher checksum posted", spec
        if "MISMATCH" in status or status == "not in publisher list":
            with open(os.path.join(DATA, f), "rb") as fh:
                if fh.read(200).lstrip().lower().startswith(b"<html") and size < 2000:
                    status = "download failed: the server's 404 page was saved instead"
        rows.append({"dataset": ds, "file": f, "bytes": size, "sha256": sha, "status": status, "publisher_path": ref})
    d = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    text = "".join(f"{r.sha256}  data/{r.file}\n" for r in d.itertuples())
    with open("results/data_sha256sums.txt.gz", "wb") as fo, gzip.GzipFile(fileobj=fo, mode="wb", mtime=0) as g:
        g.write(text.encode())
    summary = d.pivot_table(index="dataset", columns="status", values="file", aggfunc="count", fill_value=0)
    summary.insert(0, "GB", (d.groupby("dataset").bytes.sum() / 1e9).round(2))
    summary.insert(0, "files", d.groupby("dataset").size())
    summary.insert(0, "source", [SOURCES[k][0] + ": " + (SOURCES[k][1] if isinstance(SOURCES[k][1], str)
                                                         else ", ".join(map(str, SOURCES[k][1]))) for k in summary.index])
    for h in HOLTER:
        summary.loc[h, "source"] = "PhysioNet via wfdb (streamed; only .npy caches on disk, not checkable)"
    summary.to_csv("results/data_manifest_summary.csv")
    ok = d.status.isin(["publisher SHA-256 match", "publisher MD5 match", "unpacked: matches archive member"])
    d[~ok].drop(columns="sha256").to_csv("results/data_manifest_problems.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string())
    print(f"\nverified against the publisher (directly or via its archive): {ok.sum()} of {len(d)} files")
