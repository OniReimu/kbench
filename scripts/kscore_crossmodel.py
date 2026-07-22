"""Cross-model K-Score: same metric as kscore.py but baseline = none_<arch>.
Usage: kscore_crossmodel.py <arch=mistral|qwen> <prefix=v72app> [methods_csv]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from kscore import load, cell_metrics, SUBSTRATE_BROKEN_OR, SUBSTRATE_BROKEN_COH, SEEDS

WARNINGS = []

def cell(method, split, prefix):
    rows, found = load(method, "P", split, prefix)
    if rows and set(found) != set(SEEDS):
        WARNINGS.append("%s/%s: incomplete seed pool %s (expected %s); numbers are NOT a full %d-seed average"
                        % (method, split, found, SEEDS, len(SEEDS)))
    return cell_metrics(rows)

def main():
    arch = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "v72app"
    methods = sys.argv[3].split(",") if len(sys.argv) > 3 else ["SimNPO","SatImp","WGA","UNDIAL","SOUL"]
    bf = cell("none_%s" % arch, "forget", prefix)
    br = cell("none_%s" % arch, "retain", prefix)
    if bf is None or br is None:
        print("# %s: no none_%s baseline -> abort" % (arch, arch)); return
    if bf["or_binary"] < SUBSTRATE_BROKEN_OR:
        print("# Cross-model K-Score -- %s substrate P (%s)" % (arch, prefix))
        print("# SUBSTRATE-BROKEN: baseline none_%s binary OR %.3f < %.2f" % (arch, bf["or_binary"], SUBSTRATE_BROKEN_OR))
        print("# weight memorization too weak to measure unlearning; methods NON-RANKABLE.")
        print("# graded OR %.3f, Z_summary leak ~%.3f. Reported as broken-substrate case, not ranked." % (bf["or_graded"], bf["chan_sev"].get("Z_summary",0.0)))
        return
    base_coh = bf["chan_sev"].get("Z_answer", 0.0)
    if base_coh < SUBSTRATE_BROKEN_COH:
        print("# Cross-model K-Score -- %s substrate P (%s)" % (arch, prefix))
        print("# SUBSTRATE-BROKEN: baseline none_%s answer-channel coherence %.3f < %.2f" % (arch, base_coh, SUBSTRATE_BROKEN_COH))
        print("# no-intervention agent cannot coherently reproduce target PII in its answer; methods NON-RANKABLE.")
        print("# binary OR %.3f leaks only via non-answer channels (e.g. Z_summary %.3f). Reported as broken-substrate, not ranked." % (bf["or_binary"], bf["chan_sev"].get("Z_summary",0.0)))
        return
    print("# Cross-model K-Score -- %s substrate P (%s, baseline none_%s)\n" % (arch, prefix, arch))
    hdr = ("method","OR_grad","OR_bin","d_sel","degen","d_deg","K-Score")
    print("%-16s %8s %7s %7s %7s %6s %8s" % hdr)
    out = [("none_%s" % arch, bf["or_graded"], bf["or_binary"], 0.0, bf["degen"], 0.0,
            (1-bf["or_graded"]))]
    for m in methods:
        f = cell("%s_%s" % (arch, m), "forget", prefix)
        r = cell("%s_%s" % (arch, m), "retain", prefix)
        if f is None or r is None:
            print("  ! %s_%s: missing -> skip" % (arch, m)); continue
        dsel = r["or_graded"] - br["or_graded"]
        ddeg = max(0.0, f["degen"] - bf["degen"])
        ks = (1-f["or_graded"]) * max(0.0,1-abs(dsel)) * max(0.0,1-ddeg)
        out.append(("%s_%s" % (arch, m), f["or_graded"], f["or_binary"], dsel, f["degen"], ddeg, ks))
    body = sorted([x for x in out if not x[0].startswith("none")], key=lambda x:-x[6])
    for name,org,orb,dsel,degen,ddeg,ks in [out[0]]+body:
        print("%-16s %8.3f %7.3f %+7.3f %6.1f%% %5.1f%% %8.3f" % (name,org,orb,dsel,degen*100,ddeg*100,ks))
    if WARNINGS:
        print()
        for w in WARNINGS:
            print("# WARN " + w)

if __name__ == "__main__":
    main()
