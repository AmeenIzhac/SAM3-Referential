#!/usr/bin/env python3
"""Download the 2,404 unique COCO train2017 images referenced by MMR_val.json."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request

VAL = "/mnt/data0/ameen/mmr_data/MMR_val.json"
OUT = "/mnt/data0/ameen/mmr_data/images"

d = json.load(open(VAL))
jobs = {}
for r in d:
    jobs[r["file_name"]] = r["coco_url"]   # e.g. train2017/xxx.jpg -> url
print(f"{len(jobs)} unique images to fetch", flush=True)

def fetch(item):
    fn, url = item
    dst = os.path.join(OUT, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, dst)
            return "ok"
        except Exception as e:
            if attempt == 3:
                return f"FAIL {fn}: {e}"

ok = skip = fail = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, res in enumerate(ex.map(fetch, jobs.items()), 1):
        if res == "ok": ok += 1
        elif res == "skip": skip += 1
        else:
            fail += 1; print(res, flush=True)
        if i % 300 == 0:
            print(f"  {i}/{len(jobs)}  ok={ok} skip={skip} fail={fail}", flush=True)
print(f"DONE ok={ok} skip={skip} fail={fail}", flush=True)
