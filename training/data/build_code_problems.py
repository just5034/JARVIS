"""Build the HEP code problem set for trace generation (Phase 4C-new).

Combines multiple sources into a single JSONL of code problems that the
teacher model will solve to produce reasoning + code traces. The output
feeds the hep_code adapter SFT.

Sources:
  1. Curated HEP-flavored coding problems (Geant4, Pythia8, ROOT, GDML)
  2. GRACE-extracted code problems (configs/grace_tools.yaml, tool wrappers)
  3. LiveCodeBench general-coding subset (sanity ballast — keeps the
     adapter from drifting on non-HEP code)

Output schema (JSONL):
    {"id": str, "problem": str, "domain": str, "difficulty": str,
     "source": str, "language": str}

Usage:
    python -m training.data.build_code_problems \\
        --benchmarks-dir /work/hdd/bgde/jhill5/data/benchmarks \\
        --grace-hep-jsonl /work/hdd/bgde/jhill5/data/hep_code_problems.jsonl \\
        --output /work/hdd/bgde/jhill5/data/code_problems.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Curated HEP coding problems — small canonical seed set covering each
# tool the hep_code adapter is expected to fluently produce.
CURATED_PROBLEMS: list[dict] = [
    # --- GDML geometry generation ---
    {
        "problem": (
            "Write a Python function `make_calorimeter_gdml(layers, "
            "absorber_thickness_mm, scint_thickness_mm, output_path)` that "
            "emits a valid Geant4 GDML file describing a sampling calorimeter "
            "with `layers` alternating absorber/active layers. Use lead as "
            "the absorber and polystyrene scintillator as the active material. "
            "Verify the output by checking that <materials> contains the "
            "required NIST refs and <structure> nests volumes correctly."
        ),
        "domain": "gdml_geometry",
        "difficulty": "medium",
        "language": "python",
    },
    {
        "problem": (
            "Implement `cylindrical_tracker_gdml(name, radius_mm, length_mm, "
            "n_layers, output_dir)` that writes a GDML file describing a "
            "cylindrical silicon tracker with `n_layers` concentric tubes "
            "of equally spaced radii, all centered on the world origin. The "
            "active material is silicon. Include a wrapping mother volume "
            "filled with vacuum."
        ),
        "domain": "gdml_geometry",
        "difficulty": "medium",
        "language": "python",
    },
    # --- Geant4 ---
    {
        "problem": (
            "Write a complete Geant4 macro that runs a particle gun firing "
            "1000 muons of 5 GeV at theta=10 deg, phi=0, into a geometry "
            "loaded from 'detector.gdml'. Enable the FTFP_BERT physics list "
            "and write the per-event energy deposit to 'output.root'."
        ),
        "domain": "geant4_code",
        "difficulty": "easy",
        "language": "geant4_macro",
    },
    {
        "problem": (
            "Implement a Geant4 SteppingAction subclass in C++ that records, "
            "for every step in any volume named 'PMT_*', the global time "
            "and (x, y, z) of the step. Provide the corresponding header "
            "file and explain how to register the action with the run "
            "manager."
        ),
        "domain": "geant4_code",
        "difficulty": "hard",
        "language": "cpp",
    },
    # --- Pythia8 ---
    {
        "problem": (
            "Write a Pythia8 .cmnd configuration that generates 50000 "
            "Z+jets events at 13 TeV using NNPDF3.0 NLO PDFs, with the Z "
            "decaying to e+e- only, and writes HepMC2 output. Set a fixed "
            "seed for reproducibility."
        ),
        "domain": "pythia8_code",
        "difficulty": "easy",
        "language": "pythia8_config",
    },
    {
        "problem": (
            "Write a Python wrapper that takes a high-level dict like "
            "{'process': 'pp > t tbar', 'sqrt_s_gev': 13000, 'n_events': "
            "10000, 'pdf': 'NNPDF31_lo_as_0118', 'seed': 1} and produces a "
            "valid Pythia8 .cmnd file. Include sane defaults for parton "
            "shower, MPI, and hadronization, and validate that the process "
            "string contains exactly one '>' separator."
        ),
        "domain": "pythia8_code",
        "difficulty": "medium",
        "language": "python",
    },
    # --- ROOT analysis ---
    {
        "problem": (
            "Write a PyROOT script that opens 'events.root', reads the "
            "tree 'Delphes', and produces a 1D histogram of jet pT for "
            "all jets with |eta| < 2.5 and pT > 30 GeV. Save the histogram "
            "to 'jet_pt.root' and also export a Matplotlib PNG via "
            "uproot+matplotlib for inclusion in a paper."
        ),
        "domain": "root_analysis",
        "difficulty": "easy",
        "language": "python",
    },
    {
        "problem": (
            "Use RooFit to perform a binned maximum-likelihood fit of a "
            "Gaussian signal plus exponential background to a mass spectrum "
            "in the range [60, 120] GeV. Extract the signal yield and its "
            "uncertainty, and produce a publication-quality plot with the "
            "data, total fit, and individual components."
        ),
        "domain": "root_analysis",
        "difficulty": "hard",
        "language": "python",
    },
    # --- Delphes ---
    {
        "problem": (
            "Write a Delphes detector card describing an ATLAS-like "
            "detector. Include the tracker (sigma_pT/pT = 0.005 + "
            "0.0001*pT), ECAL (sigma_E/E = 0.10/sqrt(E) + 0.01), HCAL "
            "(sigma_E/E = 0.50/sqrt(E) + 0.05), muon system, and a "
            "b-tagging efficiency block parameterized in (pT, eta)."
        ),
        "domain": "delphes_code",
        "difficulty": "hard",
        "language": "delphes_card",
    },
    # --- ROOT/Pythia/Delphes pipeline ---
    {
        "problem": (
            "Write a bash + Python pipeline that (1) runs Pythia8 to "
            "produce 100k Z+jets events at 13 TeV in HepMC2 format, "
            "(2) feeds the HepMC2 to Delphes with an ATLAS card, "
            "(3) reads the Delphes output with PyROOT and produces a "
            "histogram of dilepton invariant mass for events with two "
            "isolated muons. Use stable file naming and exit on any "
            "non-zero return code."
        ),
        "domain": "hep_pipeline",
        "difficulty": "hard",
        "language": "shell+python",
    },
    # --- FastJet ---
    {
        "problem": (
            "Write a Python script using PyFastJet that reads jet "
            "constituents from a CSV (columns: event_id, px, py, pz, E), "
            "clusters them with the anti-kt algorithm at R=0.4, and "
            "writes back per-jet (event_id, jet_pT, jet_eta, jet_phi, "
            "jet_mass, n_constituents) as a Parquet file."
        ),
        "domain": "fastjet_code",
        "difficulty": "medium",
        "language": "python",
    },
    # --- Provenance / GRACE tool interface ---
    {
        "problem": (
            "Implement a Python class `Tool` with abstract methods `run("
            "self, config)`, `validate_output(self, result)`, and "
            "`manifest(self) -> dict`. The class enforces that every "
            "concrete subclass produces a content-addressed output "
            "directory named by SHA256 of the JSON-serialized config, "
            "writes a manifest.json with provenance, and emits structured "
            "JSON logs to stderr."
        ),
        "domain": "hep_pipeline",
        "difficulty": "hard",
        "language": "python",
    },
]


def load_grace_hep_code(grace_jsonl: Path) -> list[dict]:
    """Load HEP code problems extracted from GRACE."""
    if not grace_jsonl.exists():
        print(f"  WARNING: GRACE HEP code data not found at {grace_jsonl}")
        return []

    problems: list[dict] = []
    with open(grace_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problems.append({
                "id": f"grace_code_{len(problems)}",
                "problem": row["problem"],
                "domain": row.get("domain", "hep_pipeline"),
                "difficulty": row.get("difficulty", "hard"),
                "source": row.get("source", "grace/hep_code"),
                "language": row.get("language", "python"),
            })
    return problems


def load_livecodebench_subset(benchmarks_dir: Path, max_count: int = 200) -> list[dict]:
    """Load a small LiveCodeBench subset as general-coding ballast.

    The hep_code adapter must remain competent at non-HEP code, otherwise
    we degrade JARVIS's general code score. A modest LCB tail keeps the
    distribution honest.
    """
    # Canonical path used by training.eval.run_livecode and download_benchmarks.
    lcb_path = benchmarks_dir / "livecode" / "livecode_bench.jsonl"
    if not lcb_path.exists():
        print(f"  WARNING: LiveCodeBench data not found at {lcb_path}")
        return []

    problems: list[dict] = []
    with open(lcb_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # download_benchmarks normalizes question_content -> description.
            problem_text = (
                row.get("problem")
                or row.get("description")
                or row.get("question_content")
                or ""
            )
            if not problem_text:
                continue
            problems.append({
                "id": f"lcb_{len(problems)}",
                "problem": problem_text,
                "domain": "general_code",
                "difficulty": row.get("difficulty", "medium"),
                "source": "livecodebench",
                "language": row.get("language", "python"),
            })
            if len(problems) >= max_count:
                break
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HEP code problem set for trace generation")
    parser.add_argument(
        "--benchmarks-dir",
        default="/work/hdd/bgde/jhill5/data/benchmarks",
        help="Directory containing downloaded benchmarks",
    )
    parser.add_argument(
        "--grace-hep-jsonl",
        default="/work/hdd/bgde/jhill5/data/hep_code_problems.jsonl",
        help="Path to GRACE-extracted HEP code problems JSONL "
             "(produced by training.data.extract_hep_code).",
    )
    parser.add_argument(
        "--output",
        default="/work/hdd/bgde/jhill5/data/code_problems.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--lcb-count",
        type=int,
        default=200,
        help="How many LiveCodeBench problems to include as general-code ballast.",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    all_problems: list[dict] = []

    # 1. Curated HEP coding problems
    print("[code-problems] adding curated HEP coding problems...")
    for i, p in enumerate(CURATED_PROBLEMS):
        record = dict(p)
        record["id"] = f"curated_code_{i}"
        record["source"] = "curated_hep"
        all_problems.append(record)
    print(f"  ->{len(CURATED_PROBLEMS)} curated problems")

    # 2. GRACE-extracted HEP code
    print("[code-problems] loading GRACE HEP code...")
    grace = load_grace_hep_code(Path(args.grace_hep_jsonl))
    all_problems.extend(grace)
    print(f"  ->{len(grace)} GRACE HEP code problems")

    # 3. LiveCodeBench ballast
    print(f"[code-problems] loading LiveCodeBench subset (max {args.lcb_count})...")
    lcb = load_livecodebench_subset(benchmarks_dir, max_count=args.lcb_count)
    all_problems.extend(lcb)
    print(f"  ->{len(lcb)} LiveCodeBench problems")

    # Backfill IDs.
    for i, p in enumerate(all_problems):
        if "id" not in p:
            p["id"] = f"code_prob_{i}"

    # Save.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Summary.
    sources = Counter(p.get("source", "unknown") for p in all_problems)
    domains = Counter(p.get("domain", "unknown") for p in all_problems)
    difficulties = Counter(p.get("difficulty", "unknown") for p in all_problems)
    print(f"\n[code-problems] total: {len(all_problems)} code problems")
    print("  by source:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")
    print("  by domain:")
    for d, count in sorted(domains.items()):
        print(f"    {d}: {count}")
    print("  by difficulty:")
    for d, count in sorted(difficulties.items()):
        print(f"    {d}: {count}")
    print(f"  saved to: {output_path}")


if __name__ == "__main__":
    main()
