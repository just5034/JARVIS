"""Quick quality checks on filtered traces before SFT.

Usage:
    python scripts/check_traces.py /work/hdd/bgde/jhill5/data/physics_filtered.jsonl
"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/work/hdd/bgde/jhill5/data/physics_filtered.jsonl"

domains = Counter()
short = 0
empty_trace = 0
no_think_tag = 0
lengths = []
token_counts = []

with open(path) as f:
    for line in f:
        d = json.loads(line)
        trace = d.get("trace", "")
        domains[d.get("domain", "unknown")] += 1
        tlen = len(trace)
        lengths.append(tlen)
        token_counts.append(d.get("tokens", 0))
        if not trace:
            empty_trace += 1
        if tlen < 100:
            short += 1
        if "<think>" not in trace:
            no_think_tag += 1

lengths.sort()
token_counts.sort()
n = len(lengths)

print(f"=== Trace Quality Report ({n} traces) ===\n")

print("Content checks:")
print(f"  Empty traces:       {empty_trace}")
print(f"  Very short (<100c): {short}")
print(f"  Missing <think> tag: {no_think_tag}")
print()

print("Length distribution (chars):")
print(f"  Min: {lengths[0]:,}  P10: {lengths[n//10]:,}  Median: {lengths[n//2]:,}  P90: {lengths[9*n//10]:,}  Max: {lengths[-1]:,}")
print()

print("Token distribution:")
print(f"  Min: {token_counts[0]:,}  P10: {token_counts[n//10]:,}  Median: {token_counts[n//2]:,}  P90: {token_counts[9*n//10]:,}  Max: {token_counts[-1]:,}")
print()

print("Domain distribution:")
for d, c in domains.most_common():
    print(f"  {d}: {c}")
