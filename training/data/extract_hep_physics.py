"""Extract HEP physics problems from the GRACE repository.

Produces a JSONL of {problem, domain, difficulty, source} records for the
hep_physics LoRA training pipeline. The teacher model generates responses
in Phase 3 (rejection sampling); this script only emits problem prompts.

Sources mined:
  1. GRACE knowledge graphs (src/grace/knowledge/*.py)
       - scintillator_kg.py, detector_geometry_kg.py, gluex_kg.py,
         higgs_tau_tau_kg.py, reasoning_kg.py, constraint_provider.py
  2. GRACE benchmark papers (data/papers/benchmark_papers.json)
  3. GRACE example experiment YAMLs (configs/example_experiment.yaml)

Output schema (JSONL):
    {"problem": str, "domain": str, "difficulty": str, "source": str}

Domains emitted:
    detector_physics, scintillator_physics, calorimeter_design,
    photon_detection, kinematics, particle_physics, hep_experiment,
    statistical_mechanics

Usage:
    python -m training.data.extract_hep_physics \\
        --grace-repo /path/to/Generative-Retrained-Agentic-Control-Environment \\
        --output /work/hdd/bgde/jhill5/data/hep_physics_problems.jsonl
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

logger = logging.getLogger("extract_hep_physics")


# ---------------------------------------------------------------------------
# Problem record
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    problem: str
    domain: str
    difficulty: str  # "easy" | "medium" | "hard"
    source: str

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Knowledge-graph parsing
# ---------------------------------------------------------------------------

def _load_module_dict(py_path: Path) -> dict[str, object]:
    """Parse a Python KG module and return its top-level dict/list literals.

    GRACE knowledge graphs store data as module-level dict/list assignments
    (e.g., MATERIALS = {...}). We use ast to extract them safely without
    importing GRACE.
    """
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    # Skip non-literal assignments (function calls, dataclass
                    # construction, etc.). We only want pure data tables.
                    continue
    return out


def extract_scintillator_problems(kg_path: Path) -> list[Problem]:
    """Generate problems from scintillator_kg.py.

    For each detector medium with a MaterialProperties definition, emit:
      - 1 lookup problem (easy)
      - 1 calculation problem (medium)
      - 1 design-tradeoff problem (hard)
    """
    if not kg_path.exists():
        logger.warning("scintillator KG not found at %s", kg_path)
        return []

    problems: list[Problem] = []
    text = kg_path.read_text(encoding="utf-8")

    # Materials are defined as @classmethod factories on MaterialProperties.
    # Pattern: `def water(cls) -> "MaterialProperties":` — extract the method
    # name and humanize it. This avoids PhotosensorSpec name= leakage.
    factory_names = re.findall(
        r'def\s+(\w+)\s*\(\s*cls\s*\)\s*->\s*["\']MaterialProperties["\']\s*:',
        text,
    )
    material_names = sorted(set(n.replace("_", " ") for n in factory_names))

    for name in material_names[:25]:
        problems.append(Problem(
            problem=(
                f"What is the typical scintillation yield (photons/MeV) and "
                f"refractive index of {name}? Give the values you would use "
                f"to estimate detector light yield for a 1 MeV deposition."
            ),
            domain="scintillator_physics",
            difficulty="easy",
            source="grace/scintillator_kg",
        ))
        problems.append(Problem(
            problem=(
                f"A 10-meter spherical vessel filled with {name} is instrumented "
                f"with PMTs at 35% photocathode coverage and 25% quantum efficiency. "
                f"Estimate the number of detected photoelectrons for a 5 MeV "
                f"electron event at the center of the vessel. State all attenuation "
                f"and geometric assumptions."
            ),
            domain="photon_detection",
            difficulty="medium",
            source="grace/scintillator_kg",
        ))
        problems.append(Problem(
            problem=(
                f"Compare {name} against liquid argon (LAr) for a low-energy "
                f"neutrino detector requiring sub-nanosecond timing resolution. "
                f"Discuss tradeoffs in scintillation yield, attenuation length, "
                f"emission spectrum, and PMT/SiPM compatibility. Recommend which "
                f"medium is appropriate and justify with quantitative estimates."
            ),
            domain="detector_physics",
            difficulty="hard",
            source="grace/scintillator_kg",
        ))

    logger.info("scintillator_kg: %d problems", len(problems))
    return problems


def extract_geometry_problems(kg_path: Path) -> list[Problem]:
    """Generate detector-geometry problems from detector_geometry_kg.py."""
    if not kg_path.exists():
        logger.warning("geometry KG not found at %s", kg_path)
        return []

    text = kg_path.read_text(encoding="utf-8")

    # Pull out topology and calorimeter-type enum members.
    topologies = re.findall(r'TopologyType\.([A-Z_]+)', text)
    cal_types = re.findall(r'CalorimeterType\.([A-Z_]+)', text)

    topo_set = sorted(set(topologies))
    cal_set = sorted(set(cal_types))

    problems: list[Problem] = []
    for topo in topo_set:
        topo_name = topo.lower().replace("_", " ")
        problems.append(Problem(
            problem=(
                f"For a {topo_name} calorimeter geometry, list the three dominant "
                f"physics tradeoffs (energy resolution, hermeticity, longitudinal "
                f"segmentation) and give a representative parameter range you "
                f"would target for a 100 GeV electron sample."
            ),
            domain="calorimeter_design",
            difficulty="medium",
            source="grace/detector_geometry_kg",
        ))

    for cal in cal_set:
        cal_name = cal.lower().replace("_", " ")
        problems.append(Problem(
            problem=(
                f"Design a {cal_name} calorimeter targeting σ/E = 10%/√E ⊕ 1% "
                f"for electromagnetic showers. Specify the absorber material, "
                f"sampling fraction, total radiation lengths, and Molière radius "
                f"considerations. Justify each choice from first principles."
            ),
            domain="calorimeter_design",
            difficulty="hard",
            source="grace/detector_geometry_kg",
        ))

    # Cross product: a few representative topology × material design problems.
    for topo in topo_set[:4]:
        for absorber in ["lead", "tungsten", "iron", "copper"]:
            problems.append(Problem(
                problem=(
                    f"You are tasked with constructing a {topo.lower()} calorimeter "
                    f"with {absorber} absorber and plastic scintillator active layers. "
                    f"Compute the radiation length, Molière radius, and shower "
                    f"containment radius for 50 GeV electrons. Recommend a layer "
                    f"thickness in units of X₀."
                ),
                domain="calorimeter_design",
                difficulty="hard",
                source="grace/detector_geometry_kg",
            ))

    logger.info("detector_geometry_kg: %d problems", len(problems))
    return problems


def extract_gluex_problems(kg_path: Path) -> list[Problem]:
    """Generate GlueX-specific particle physics problems."""
    if not kg_path.exists():
        return []

    problems = [
        Problem(
            problem=(
                "The GlueX experiment at Jefferson Lab studies photoproduction of "
                "exotic mesons. For a 9 GeV linearly polarized photon beam on a "
                "liquid hydrogen target, estimate the production cross-section "
                "for the η(958) and discuss the kinematic acceptance of the BCAL "
                "and FCAL calorimeters."
            ),
            domain="hep_experiment",
            difficulty="hard",
            source="grace/gluex_kg",
        ),
        Problem(
            problem=(
                "Explain the role of the Central Drift Chamber (CDC) in GlueX. "
                "Given typical operating parameters (4 atm gas, 10 mm cell size), "
                "estimate the position resolution achievable for a 1 GeV charged "
                "track and the maximum drift time."
            ),
            domain="detector_physics",
            difficulty="medium",
            source="grace/gluex_kg",
        ),
        Problem(
            problem=(
                "Predict the angular distribution dN/dcos(θ_GJ) for the decay "
                "π₁(1600) → ρπ in the Gottfried-Jackson frame, assuming the "
                "exotic state is produced via natural-parity exchange. Sketch "
                "what the distribution looks like and identify the diagnostic "
                "feature that distinguishes J^PC = 1^-+ from 1^++."
            ),
            domain="particle_physics",
            difficulty="hard",
            source="grace/gluex_kg",
        ),
    ]
    logger.info("gluex_kg: %d problems", len(problems))
    return problems


def extract_higgs_problems(kg_path: Path) -> list[Problem]:
    """Generate Higgs analysis problems from higgs_tau_tau_kg.py."""
    if not kg_path.exists():
        return []

    problems = [
        Problem(
            problem=(
                "In an H→ττ analysis, the visible mass m_vis is biased low because "
                "the neutrinos from τ decays carry undetected momentum. Derive "
                "the missing-mass calculator (MMC) approach and explain why the "
                "collinear approximation fails near m_H = 125 GeV. What is the "
                "expected m_ττ resolution after MMC reconstruction?"
            ),
            domain="hep_experiment",
            difficulty="hard",
            source="grace/higgs_tau_tau_kg",
        ),
        Problem(
            problem=(
                "Given the Higgs ML challenge feature set (PRI_tau_pt, "
                "PRI_lep_pt, DER_mass_MMC, etc.), propose a minimal feature "
                "subset that preserves > 95% of the AMS metric, and justify "
                "your choice using physics rather than statistics."
            ),
            domain="hep_experiment",
            difficulty="medium",
            source="grace/higgs_tau_tau_kg",
        ),
    ]
    logger.info("higgs_tau_tau_kg: %d problems", len(problems))
    return problems


def extract_constraint_problems(kg_path: Path) -> list[Problem]:
    """Generate physics-bounds problems from constraint_provider.py."""
    if not kg_path.exists():
        return []

    text = kg_path.read_text(encoding="utf-8")
    # Find PhysicsBounds field names referenced as "lookup_*"
    field_names = sorted(set(re.findall(r'["\']([a-z_]+_(?:length|radius|yield|ratio|index|peak))["\']', text)))

    problems: list[Problem] = []
    for field in field_names[:15]:
        nice = field.replace("_", " ")
        problems.append(Problem(
            problem=(
                f"State the literature reference value (with units and "
                f"uncertainty) for {nice} in liquid argon, water-Cherenkov, and "
                f"plastic scintillator. For each, identify the dominant "
                f"systematic uncertainty in a real-world measurement."
            ),
            domain="detector_physics",
            difficulty="medium",
            source="grace/constraint_provider",
        ))

    logger.info("constraint_provider: %d problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# YAML / paper sources
# ---------------------------------------------------------------------------

def extract_experiment_yaml_problems(yaml_path: Path) -> list[Problem]:
    """Generate analysis-design problems from example_experiment.yaml."""
    if not yaml_path.exists():
        return []
    try:
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s: %s", yaml_path, exc)
        return []

    process = (spec.get("generation") or {}).get("process", "pp > Z' > t tbar")
    cuts = (spec.get("analysis") or {}).get("event_selection", [])
    observables = (spec.get("analysis") or {}).get("observables", [])

    problems = [
        Problem(
            problem=(
                f"For the process {process} at √s = 13 TeV, estimate the leading-"
                f"order cross-section, identify the dominant background, and "
                f"propose three discriminating variables. Sketch the signal-to-"
                f"background scaling as a function of an integrated luminosity "
                f"of 30 fb⁻¹ to 3000 fb⁻¹."
            ),
            domain="hep_experiment",
            difficulty="hard",
            source="grace/example_experiment",
        ),
    ]
    if cuts:
        problems.append(Problem(
            problem=(
                f"Given the event selection {cuts!r}, justify each cut from a "
                f"physics standpoint and estimate the cut efficiency on signal "
                f"and on the dominant background. Recommend one cut to tighten "
                f"and one to loosen, with quantitative justification."
            ),
            domain="hep_experiment",
            difficulty="medium",
            source="grace/example_experiment",
        ))
    if observables:
        problems.append(Problem(
            problem=(
                f"For the observables {observables!r}, propose a binning scheme "
                f"appropriate for an integrated luminosity of 300 fb⁻¹. State "
                f"the resolution model you assume for each observable and how "
                f"it sets the minimum bin width."
            ),
            domain="hep_experiment",
            difficulty="medium",
            source="grace/example_experiment",
        ))
    logger.info("example_experiment.yaml: %d problems", len(problems))
    return problems


def extract_paper_problems(papers_json: Path) -> list[Problem]:
    """Generate problems from the curated benchmark_papers.json."""
    if not papers_json.exists():
        return []
    try:
        data = json.loads(papers_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", papers_json, exc)
        return []

    papers = data if isinstance(data, list) else data.get("papers", [])
    problems: list[Problem] = []
    for entry in papers:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title") or entry.get("name") or "the reference paper"
        detector = entry.get("detector") or entry.get("experiment") or "the detector"
        problems.append(Problem(
            problem=(
                f"Reproduce the principal result of '{title}' for the {detector} "
                f"experiment. Identify the input assumptions, the analysis "
                f"chain, and the key systematic uncertainties. Estimate how the "
                f"result would change if the photocathode coverage were halved."
            ),
            domain="hep_experiment",
            difficulty="hard",
            source=f"grace/benchmark_papers:{detector}",
        ))
    logger.info("benchmark_papers.json: %d problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect(grace_repo: Path) -> list[Problem]:
    kg = grace_repo / "src" / "grace" / "knowledge"
    configs = grace_repo / "configs"
    papers = grace_repo / "data" / "papers" / "benchmark_papers.json"

    problems: list[Problem] = []
    problems += extract_scintillator_problems(kg / "scintillator_kg.py")
    problems += extract_geometry_problems(kg / "detector_geometry_kg.py")
    problems += extract_gluex_problems(kg / "gluex_kg.py")
    problems += extract_higgs_problems(kg / "higgs_tau_tau_kg.py")
    problems += extract_constraint_problems(kg / "constraint_provider.py")
    problems += extract_experiment_yaml_problems(configs / "example_experiment.yaml")
    problems += extract_paper_problems(papers)
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
    parser = argparse.ArgumentParser(description="Extract HEP physics problems from GRACE.")
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

    # Summary by domain.
    by_domain: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for p in problems:
        by_domain[p.domain] = by_domain.get(p.domain, 0) + 1
        by_difficulty[p.difficulty] = by_difficulty.get(p.difficulty, 0) + 1
    logger.info("By domain: %s", sorted(by_domain.items()))
    logger.info("By difficulty: %s", sorted(by_difficulty.items()))


if __name__ == "__main__":
    main()
