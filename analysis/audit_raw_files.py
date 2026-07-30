"""
audit_raw_files.py - integrity audit of the raw .dat files.

Examines exactly the set of files the pipeline sees (as iter_all_events does):
  1. Byte-identical files (md5), for example the pair 210206.dat / 210206-c.dat.
  2. Events duplicated between files, matched on the full |EVENT: header line,
     and within a single file, giving the inflation of the total event count.
  3. Files whose events do not belong to the year of their directory
     (for example 220712.dat sitting in 2021/).
  4. Irregular filenames, not matching the six-digit YYMMDD.dat pattern: -c, _0
     and x- variants, and files from which no event could be parsed.
  5. .dat files inside raw_dir that the scan never reaches, that is outside the
     year directories.
  6. A ready block of recommendations for data.exclude_files.

Nothing is written to disk. The short report is wrapped in a PASTE-BLOCK; the
full per-file table is printed with --full and is best redirected to a file:
    ... --full > output/paper_plots/audit_full.txt

Run from the project root:
    python analysis/audit_raw_files.py --config config/settings.yaml
    python analysis/audit_raw_files.py --raw-dir data/raw/bank0 --years 2021 2023 --no-md5
"""
import os
import re
import sys
import glob
import time
import hashlib
import argparse
from datetime import datetime
from collections import defaultdict, Counter

REGULAR_NAME = re.compile(r"^\d{6}\.dat$")


def gather_files_like_pipeline(raw_dir, years):
    """The same file order and selection as data_loader.iter_all_events."""
    files = []
    if years:
        for year in years:
            ydir = os.path.join(raw_dir, str(year))
            if os.path.isdir(ydir):
                files.extend(sorted(glob.glob(os.path.join(ydir, "*.dat"))))
    else:
        files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.dat"), recursive=True))
    return files


def scan_file(path, do_md5=True):
    """Read the file once: md5 and event headers, without parsing the arrays."""
    with open(path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest() if do_md5 else "-"
    text = data.decode("utf-8", errors="ignore")

    headers = []          # full header lines, in file order
    bad_headers = 0
    tmin = tmax = None
    year_counter = Counter()

    for line in text.splitlines():
        if not line.startswith("|EVENT: "):
            continue
        line = line.strip()
        parts = line.split()
        try:
            dt = datetime.strptime(parts[1] + " " + parts[2], "%d.%m.%Y %H:%M:%S")
        except (IndexError, ValueError):
            bad_headers += 1
            continue  # the pipeline skips such events as well
        headers.append(line)
        year_counter[dt.year] += 1
        if tmin is None or dt < tmin:
            tmin = dt
        if tmax is None or dt > tmax:
            tmax = dt

    return {
        "path": path,
        "name": os.path.basename(path),
        "size": len(data),
        "md5": md5,
        "headers": headers,
        "n_events": len(headers),
        "bad_headers": bad_headers,
        "tmin": tmin,
        "tmax": tmax,
        "years": year_counter,
    }


def folder_year(path):
    m = re.search(r"[\\/](\d{4})[\\/][^\\/]+$", path)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description="Integrity audit of the raw .dat files (duplicates, wrong season)")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--no-md5", action="store_true", help="skip md5 hashing (faster)")
    ap.add_argument("--pair-frac", type=float, default=0.5,
                    help="fraction of shared events, relative to the smaller file, that flags a pair")
    ap.add_argument("--pair-min", type=int, default=20,
                    help="minimum number of shared events that flags a pair")
    ap.add_argument("--full", action="store_true", help="print the per-file table")
    args = ap.parse_args()

    raw_dir, years = args.raw_dir, args.years
    if raw_dir is None or years is None:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            raw_dir = raw_dir or cfg["data"]["raw_dir"]
            if years is None:
                years = cfg["data"].get("years")
        except Exception as e:
            sys.exit(f"Could not read the config ({e}); pass --raw-dir and --years.")

    files = gather_files_like_pipeline(raw_dir, years)
    if not files:
        sys.exit(f"No .dat files in {raw_dir} (years={years})")

    # .dat files the pipeline never reaches (outside the year directories)
    all_dat = set(glob.glob(os.path.join(raw_dir, "**", "*.dat"), recursive=True))
    unscanned = sorted(all_dat - set(files))
    # non-.dat files inside the year directories, for information
    stray = []
    for y in (years or []):
        ydir = os.path.join(raw_dir, str(y))
        if os.path.isdir(ydir):
            for e in sorted(os.listdir(ydir)):
                if not e.endswith(".dat") and os.path.isfile(os.path.join(ydir, e)):
                    stray.append(f"{y}/{e}")

    t0 = time.time()
    infos = []
    occurrences = {}                 # header -> [file_idx, ...]
    pair_shared = defaultdict(int)   # (i, j) -> shared events
    intra_dup = Counter()            # file_idx -> duplicates within the file
    for idx, path in enumerate(files):
        info = scan_file(path, do_md5=not args.no_md5)
        infos.append(info)
        for h in info["headers"]:
            lst = occurrences.get(h)
            if lst is None:
                occurrences[h] = [idx]
            else:
                for j in lst:
                    if j == idx:
                        intra_dup[idx] += 1
                    else:
                        a, b = (j, idx) if j < idx else (idx, j)
                        pair_shared[(a, b)] += 1
                lst.append(idx)
        print(f"  scanned {idx+1:>3}/{len(files)}: {info['name']:<22} "
              f"events={info['n_events']:>6}", file=sys.stderr)

    total_events = sum(i["n_events"] for i in infos)
    distinct_events = len(occurrences)
    dup_instances = total_events - distinct_events
    total_bad = sum(i["bad_headers"] for i in infos)

    # md5 groups
    md5_groups = defaultdict(list)
    if not args.no_md5:
        for i, info in enumerate(infos):
            md5_groups[info["md5"]].append(i)
    md5_dups = {h: idxs for h, idxs in md5_groups.items() if len(idxs) > 1}

    # pairs with a large overlap
    flagged_pairs = []
    for (a, b), shared in pair_shared.items():
        na, nb = infos[a]["n_events"], infos[b]["n_events"]
        frac = shared / max(1, min(na, nb))
        if shared >= args.pair_min and frac >= args.pair_frac:
            flagged_pairs.append((frac, shared, a, b))
    flagged_pairs.sort(reverse=True)

    # season inconsistent with the directory or the config
    cfg_years = set(years or [])
    season_flags = []
    for i, info in enumerate(infos):
        fy = folder_year(info["path"])
        bad_years = {y: c for y, c in info["years"].items()
                     if (fy is not None and y != fy) or (cfg_years and y not in cfg_years)}
        if bad_years:
            season_flags.append((i, fy, bad_years))

    irregular = [i for i, info in enumerate(infos) if not REGULAR_NAME.match(info["name"])]
    empty = [i for i, info in enumerate(infos) if info["n_events"] == 0]

    # --- recommendations for exclude_files ---
    rec = {}  # name -> reason
    for _, idxs in md5_dups.items():
        keep = min(idxs, key=lambda k: (0 if REGULAR_NAME.match(infos[k]["name"]) else 1,
                                        infos[k]["name"]))
        for k in idxs:
            if k != keep:
                rec[infos[k]["name"]] = f"byte-identical copy of {infos[keep]['name']}"
    for frac, shared, a, b in flagged_pairs:
        if frac < 0.9:
            continue  # partial overlap: decide by hand
        ia, ib = infos[a], infos[b]
        ra, rb = bool(REGULAR_NAME.match(ia["name"])), bool(REGULAR_NAME.match(ib["name"]))
        if ra != rb:
            drop = ib if ra else ia
        else:
            drop = ia if ia["n_events"] <= ib["n_events"] else ib
        keep = ib if drop is ia else ia
        rec.setdefault(drop["name"],
                       f"{frac:.0%} of its events also in {keep['name']} ({shared})")
    for i, fy, bad_years in season_flags:
        if sum(bad_years.values()) == infos[i]["n_events"]:
            rec.setdefault(infos[i]["name"],
                           f"all events from {sorted(bad_years)}, not the season of directory {fy}")
    for i in empty:
        rec.setdefault(infos[i]["name"], "no events parsed")

    dt = time.time() - t0

    # ================= REPORT (PASTE-BLOCK) =================
    P = print
    P("=" * 68)
    P("===== PASTE-BLOCK AUDIT v1 START =====")
    P(f"raw_dir={raw_dir}")
    P(f"years={years}  files_scanned={len(files)}  scan_time={dt:.0f}s  "
      f"md5={'off' if args.no_md5 else 'on'}")
    P(f"TOTAL events={total_events}  distinct={distinct_events}  "
      f"DUPLICATE_INSTANCES={dup_instances}  bad_headers={total_bad}")
    P(f"intra-file duplicate events: {sum(intra_dup.values())} "
      f"(files affected: {len(intra_dup)})")
    P(f"unscanned .dat outside years dirs: {len(unscanned)}"
      + (f"  -> {', '.join(os.path.relpath(p, raw_dir) for p in unscanned[:6])}"
         + (" ..." if len(unscanned) > 6 else "") if unscanned else ""))
    if stray:
        P(f"non-.dat inside year dirs ({len(stray)}): " + ", ".join(stray[:8])
          + (" ..." if len(stray) > 8 else ""))

    P("-" * 68)
    P(f"[1] MD5-identical file groups: {len(md5_dups)}")
    for h, idxs in md5_dups.items():
        P("    " + "  ==  ".join(f"{infos[k]['name']}({infos[k]['n_events']}ev)"
                                 for k in idxs))

    P("-" * 68)
    P(f"[2] Event-overlap pairs (>= {args.pair_min} ev & >= {args.pair_frac:.0%} "
      f"of smaller): {len(flagged_pairs)}")
    for frac, shared, a, b in flagged_pairs[:20]:
        ia, ib = infos[a], infos[b]
        P(f"    {ia['name']:<20} ~ {ib['name']:<20} shared={shared:>6} "
          f"({frac:>4.0%} of min; sizes {ia['n_events']}/{ib['n_events']})")
    if len(flagged_pairs) > 20:
        P(f"    ... and {len(flagged_pairs)-20} more pairs (see --full)")

    P("-" * 68)
    P(f"[3] Files with events outside their folder-year / config years: "
      f"{len(season_flags)}")
    for i, fy, bad_years in season_flags:
        info = infos[i]
        det = ", ".join(f"{y}:{c}" for y, c in sorted(bad_years.items()))
        P(f"    {info['name']:<20} folder={fy} events={info['n_events']:>6} "
          f"offending[{det}]  span {info['tmin']}..{info['tmax']}")

    P("-" * 68)
    P(f"[4] Irregular filenames (not ^DDDDDD.dat$): {len(irregular)}")
    for i in irregular:
        info = infos[i]
        mark = " <-- EMPTY" if info["n_events"] == 0 else ""
        P(f"    {info['name']:<22} events={info['n_events']:>6} "
          f"span {info['tmin']}..{info['tmax']}{mark}")

    P("-" * 68)
    P("[5] RECOMMENDED for data.exclude_files (review before applying):")
    if rec:
        P("exclude_files:")
        for name in sorted(rec):
            P(f"  - {name}   # {rec[name]}")
    else:
        P("    (empty: no duplicates or misfiled seasons found)")
    P("===== PASTE-BLOCK AUDIT v1 END =====")
    P("=" * 68)

    if args.full:
        P("\nFULL PER-FILE TABLE:")
        P(f"{'file':<22} {'events':>7} {'bad':>4} {'size_MB':>8} "
          f"{'first_event':>19} {'last_event':>19} {'md5[:8]':>9}")
        for info in infos:
            P(f"{info['name']:<22} {info['n_events']:>7} {info['bad_headers']:>4} "
              f"{info['size']/1e6:>8.1f} {str(info['tmin']):>19} "
              f"{str(info['tmax']):>19} {info['md5'][:8]:>9}")


if __name__ == "__main__":
    main()