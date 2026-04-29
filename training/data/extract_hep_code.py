"""Extract HEP code problems from the GRACE repository.

Produces a JSONL of {problem, domain, difficulty, source} records for the
hep_code LoRA training pipeline. The teacher model generates code responses
in Phase 3 (rejection sampling); this script only emits problem prompts.

Sources mined:
  1. GRACE tool implementations (src/grace/tools/*.py)
       - geant4.py, pythia8.py, root.py, delphes.py, fastjet.py
  2. GRACE geometry generators (src/grace/tools/geometry/*.py)
       - calorimeter_gdml.py, optical_gdml.py, materials.py
  3. GRACE GDML examples (configs/detectors/*.gdml + data/external_gdml/*.gdml)
  4. GRACE tool YAMLs (configs/grace_tools.yaml)

Output schema (JSONL):
    {"problem": str, "domain": str, "difficulty": str, "source": str}

Domains emitted:
    geant4_code, pythia8_code, root_analysis, delphes_code, gdml_geometry,
    geometry_generation, hep_pipeline

Usage:
    python -m training.data.extract_hep_code \\
        --grace-repo /path/to/Generative-Retrained-Agentic-Control-Environment \\
        --output /work/hdd/bgde/jhill5/data/hep_code_problems.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("extract_hep_code")


# ---------------------------------------------------------------------------
# Problem record
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    problem: str
    domain: str
    difficulty: str
    source: str

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Tool-file inspection
# ---------------------------------------------------------------------------

def _public_functions(py_path: Path) -> list[tuple[str, str | None]]:
    """Return [(function_name, docstring), ...] for public top-level functions
    and methods of public classes.
    """
    if not py_path.exists():
        return []
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        logger.warning("Could not parse %s: %s", py_path, exc)
        return []

    out: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            out.append((node.name, ast.get_docstring(node)))
    return out


# ---------------------------------------------------------------------------
# Tool-specific extractors
# ---------------------------------------------------------------------------

def extract_geant4_problems(tool_path: Path) -> list[Problem]:
    """Generate Geant4 code problems."""
    if not tool_path.exists():
        return []

    problems = [
        Problem(
            problem=(
                "Write a Geant4 macro that fires 10000 electrons of energy 1 GeV "
                "into a calorimeter at theta=90, phi=0, and saves the energy "
                "deposit per layer to a ROOT file. Use the standard "
                "G4ParticleGun and assume the world is built from a GDML file "
                "named 'calorimeter.gdml'."
            ),
            domain="geant4_code",
            difficulty="medium",
            source="grace/tools/geant4",
        ),
        Problem(
            problem=(
                "Implement a Geant4 user action that records, for every event, "
                "the (x, y, z) position of every step that deposits more than "
                "10 keV in a sensitive detector. Output the data as a flat "
                "ROOT TTree with branches event_id, step_x, step_y, step_z, edep."
            ),
            domain="geant4_code",
            difficulty="hard",
            source="grace/tools/geant4",
        ),
        Problem(
            problem=(
                "Write a Python wrapper that converts a high-level particle "
                "specification ({'particle': 'mu-', 'energy_gev': 5.0, "
                "'direction': [0, 0, 1]}) into a Geant4 macro file. Handle "
                "particle name aliasing (e.g., 'muon' -> 'mu-', 'photon' -> "
                "'gamma') and validate that energies are positive."
            ),
            domain="geant4_code",
            difficulty="medium",
            source="grace/tools/geant4",
        ),
        Problem(
            problem=(
                "Set up a Geant4 simulation of optical photon transport through "
                "a 10 m liquid scintillator vessel with PMTs covering 30% of "
                "the inner surface. Configure the optical physics list, set "
                "the scintillation yield to 10000 photons/MeV, define the "
                "PMT photocathode as a sensitive detector, and record the "
                "photoelectron count per event."
            ),
            domain="geant4_code",
            difficulty="hard",
            source="grace/tools/geant4",
        ),
    ]

    # Mine real method names for extra problem seeds.
    for fn, doc in _public_functions(tool_path):
        if doc and len(doc) > 40 and "Geant4" in doc:
            short = doc.strip().splitlines()[0].rstrip(".")
            problems.append(Problem(
                problem=(
                    f"Implement a Python function `{fn}(...)` for the GRACE "
                    f"Geant4 tool. The function should: {short}. Include "
                    f"input validation, container-aware path handling, and "
                    f"emit structured logs for provenance."
                ),
                domain="geant4_code",
                difficulty="hard",
                source=f"grace/tools/geant4:{fn}",
            ))
    logger.info("geant4 problems: %d", len(problems))
    return problems


def extract_pythia8_problems(tool_path: Path) -> list[Problem]:
    """Generate Pythia8 problems."""
    if not tool_path.exists():
        return []

    process_strings = [
        "pp > Z > mu+ mu-",
        "pp > t tbar",
        "pp > H > b bbar",
        "pp > W+ jet",
        "pp > Z > tau+ tau-",
        "ee > Z > q qbar",
    ]

    problems: list[Problem] = []
    for process in process_strings:
        problems.append(Problem(
            problem=(
                f"Write a Pythia8 configuration that generates 100k events of "
                f"the process '{process}' at √s = 13 TeV. Use NNPDF3.0 NLO "
                f"PDFs, enable parton shower and hadronization, set a fixed "
                f"random seed of 42, and write the output to HepMC2 format."
            ),
            domain="pythia8_code",
            difficulty="medium",
            source="grace/tools/pythia8",
        ))

    problems.append(Problem(
        problem=(
            "Implement a Python helper that takes a high-level process spec "
            "(dict with 'beam', 'energy_gev', 'process', 'n_events', 'seed') "
            "and emits a complete Pythia8 .cmnd file. Validate that the "
            "process string is well-formed and that the energy is within "
            "[1, 14000] GeV."
        ),
        domain="pythia8_code",
        difficulty="medium",
        source="grace/tools/pythia8",
    ))
    problems.append(Problem(
        problem=(
            "Write a Pythia8 user hook that vetoes events where the leading "
            "lepton has |η| > 2.5 or pT < 20 GeV. Implement in C++ following "
            "the Pythia8 UserHooks interface, and provide a Python wrapper "
            "that writes a runtime-configured .cc file before compilation."
        ),
        domain="pythia8_code",
        difficulty="hard",
        source="grace/tools/pythia8",
    ))
    logger.info("pythia8 problems: %d", len(problems))
    return problems


def extract_root_problems(tool_path: Path) -> list[Problem]:
    """Generate ROOT/PyROOT analysis problems."""
    if not tool_path.exists():
        return []

    problems = [
        Problem(
            problem=(
                "Write a PyROOT analysis script that opens a TTree from "
                "events.root, applies the cuts (lepton pT > 25 GeV, "
                "MET > 30 GeV, n_jets >= 2), and produces a 1D histogram of "
                "the dilepton invariant mass with 50 bins from 60 to 120 GeV. "
                "Save the histogram to output.root."
            ),
            domain="root_analysis",
            difficulty="easy",
            source="grace/tools/root",
        ),
        Problem(
            problem=(
                "Fit a Breit-Wigner convolved with a Gaussian to a Z-boson "
                "resonance peak in m_ll between 60 and 120 GeV using ROOT's "
                "RooFit. Extract the mass, width, and Gaussian resolution. "
                "Report uncertainties from MIGRAD."
            ),
            domain="root_analysis",
            difficulty="hard",
            source="grace/tools/root",
        ),
        Problem(
            problem=(
                "Implement a PyROOT macro that compares two histograms (signal "
                "MC vs data) using a Kolmogorov-Smirnov test, plots the "
                "ratio panel below the main panel, and saves the canvas to "
                "comparison.pdf. Handle empty bins gracefully."
            ),
            domain="root_analysis",
            difficulty="medium",
            source="grace/tools/root",
        ),
        Problem(
            problem=(
                "Write a PyROOT analysis that reads a 50 GB TChain efficiently "
                "using TTreeReader (not GetEntry), processes 10 branches per "
                "event, and writes a Parquet output for downstream pandas "
                "analysis. Profile the I/O and report events/second."
            ),
            domain="root_analysis",
            difficulty="hard",
            source="grace/tools/root",
        ),
    ]
    logger.info("root problems: %d", len(problems))
    return problems


def extract_delphes_problems(tool_path: Path) -> list[Problem]:
    """Generate Delphes problems."""
    if not tool_path.exists():
        return []

    problems = [
        Problem(
            problem=(
                "Write a Delphes detector card for an ATLAS-like detector "
                "with a tracker resolution of σ(pT)/pT = 0.005 ⊕ 0.0001·pT, "
                "an ECAL resolution of σ(E)/E = 0.10/√E ⊕ 0.01, and a "
                "calorimeter eta coverage up to |η| = 4.9. Include muon "
                "reconstruction and b-tagging blocks."
            ),
            domain="delphes_code",
            difficulty="hard",
            source="grace/tools/delphes",
        ),
        Problem(
            problem=(
                "Run Delphes on a HepMC input from Pythia8 using a CMS-like "
                "card, then write a PyROOT script that reads the Delphes "
                "output and fills a histogram of reconstructed jet multiplicity "
                "for events with at least one isolated muon."
            ),
            domain="delphes_code",
            difficulty="medium",
            source="grace/tools/delphes",
        ),
    ]
    logger.info("delphes problems: %d", len(problems))
    return problems


# ---------------------------------------------------------------------------
# GDML extraction
# ---------------------------------------------------------------------------

def extract_gdml_problems(repo: Path) -> list[Problem]:
    """Generate GDML geometry problems by inspecting examples in the repo."""
    candidates = list(repo.glob("**/*.gdml")) + list(repo.glob("**/*.gdml.xml"))
    # Cap to avoid pathological scans on very deep trees.
    candidates = [p for p in candidates if "node_modules" not in p.parts][:25]

    problems: list[Problem] = []
    for gdml in candidates:
        try:
            text = gdml.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        n_box = len(re.findall(r"<box\s", text))
        n_tube = len(re.findall(r"<tube\s", text))
        n_volume = len(re.findall(r"<volume\s", text))

        descriptor = []
        if n_box: descriptor.append(f"{n_box} box solids")
        if n_tube: descriptor.append(f"{n_tube} tube solids")
        if n_volume: descriptor.append(f"{n_volume} logical volumes")
        descriptor_text = ", ".join(descriptor) or "a GDML geometry"

        problems.append(Problem(
            problem=(
                f"Write a Python function that generates a GDML file equivalent "
                f"to the example at {gdml.name}, which contains {descriptor_text}. "
                f"Parameterize the world size, the active material, and the "
                f"sensor placement so that the same code can produce both a "
                f"laboratory bench-top and a full-scale detector geometry."
            ),
            domain="gdml_geometry",
            difficulty="hard",
            source=f"grace/gdml:{gdml.name}",
        ))

    # Add canonical generation tasks per topology.
    for topo in ["box", "cylinder_barrel", "projective_tower", "shashlik", "accordion"]:
        problems.append(Problem(
            problem=(
                f"Write a Python function `generate_{topo}_gdml(name, geom, "
                f"world_size_mm, output_dir)` that emits a Geant4-readable GDML "
                f"file describing a {topo.replace('_', ' ')} calorimeter. Include "
                f"alternating absorber/active layers, a wrapping mother volume, "
                f"and material references compatible with Geant4 NIST materials."
            ),
            domain="geometry_generation",
            difficulty="hard",
            source="grace/tools/geometry",
        ))

    logger.info("gdml problems: %d (from %d files)", len(problems), len(candidates))
    return problems


# ---------------------------------------------------------------------------
# YAML tool spec extraction
# ---------------------------------------------------------------------------

def extract_grace_tools_yaml_problems(yaml_path: Path) -> list[Problem]:
    """Generate problems from configs/grace_tools.yaml tool specs."""
    if not yaml_path.exists():
        return []
    try:
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s: %s", yaml_path, exc)
        return []

    if not isinstance(spec, dict):
        return []

    problems: list[Problem] = []
    for tool_name, tool_spec in spec.items():
        if not isinstance(tool_spec, dict):
            continue
        description = tool_spec.get("description", "")
        inputs = tool_spec.get("inputs", {}) or {}
        if not description:
            continue
        input_keys = ", ".join(sorted(inputs.keys())) or "(no inputs)"
        problems.append(Problem(
            problem=(
                f"Implement a Python wrapper class `{tool_name.replace('_', '').title()}Tool` "
                f"that conforms to the GRACE Tool interface (run, validate_output, typed I/O). "
                f"Description: {description}. Inputs: {input_keys}. The wrapper "
                f"must produce a content-addressed output directory, a manifest.json "
                f"with provenance, and structured logs."
            ),
            domain="hep_pipeline",
            difficulty="hard",
            source=f"grace/tools.yaml:{tool_name}",
        ))
    logger.info("grace_tools.yaml problems: %d", len(problems))
    return problems


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect(grace_repo: Path) -> list[Problem]:
    tools = grace_repo / "src" / "grace" / "tools"
    configs = grace_repo / "configs"

    problems: list[Problem] = []
    problems += extract_geant4_problems(tools / "geant4.py")
    problems += extract_pythia8_problems(tools / "pythia8.py")
    problems += extract_root_problems(tools / "root.py")
    problems += extract_delphes_problems(tools / "delphes.py")
    problems += extract_gdml_problems(grace_repo)
    problems += extract_grace_tools_yaml_problems(configs / "grace_tools.yaml")
    return problems


def write_jsonl(problems: list[Problem], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_written = 0
    with output.open("w", encoding="utf-8") as f:
        for p in problems:
            key = p.problem.strip()
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            n_written += 1
    logger.info("Wrote %d unique problems to %s", n_written, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HEP code problems from GRACE.")
    parser.add_argument("--grace-repo", required=True, type=Path,
                        help="Path to the GRACE repository root.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSONL path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="[%(name)s] %(message)s")

    if not args.grace_repo.exists():
        raise SystemExit(f"GRACE repo not found at {args.grace_repo}")

    problems = collect(args.grace_repo)
    if not problems:
        raise SystemExit("No problems extracted; check GRACE repo path and logs.")

    write_jsonl(problems, args.output)

    by_domain: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for p in problems:
        by_domain[p.domain] = by_domain.get(p.domain, 0) + 1
        by_difficulty[p.difficulty] = by_difficulty.get(p.difficulty, 0) + 1
    logger.info("By domain: %s", sorted(by_domain.items()))
    logger.info("By difficulty: %s", sorted(by_difficulty.items()))


if __name__ == "__main__":
    main()
