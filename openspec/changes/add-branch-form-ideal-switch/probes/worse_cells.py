import json, sys, collections
sys.path[:0] = ["benchmarks/tools", "benchmarks"]
import ideal_switch_flip_evidence as ev
d = json.load(open(".scratch/t5/flip_evidence_ideal.json"))
by = {(r["circuit"], r["config"], r["variant"]): r for r in d}
out = []
for (c, cfg, v), r in by.items():
    if v not in ("ideal", "ideal+lcp"):
        continue
    dflt = by[(c, cfg, "default")]
    k = ev._cmp(dflt, r)
    if k == "worse":
        out.append((cfg, v, c, dflt["outcome"], r["outcome"], dflt.get("err_rms"), r.get("err_rms"), r.get("tol"), r.get("reference")))
cnt = collections.Counter((o[0], o[1]) for o in out)
print(dict(cnt))
for o in sorted(out):
    print(*o, sep=" | ")
