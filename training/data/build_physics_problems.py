"""Build the physics problem set for trace generation (Phase 4A).

Combines multiple sources into a single JSONL file of physics problems
that the teacher model (R1-0528) will solve to produce reasoning traces.

Sources:
  1. MMLU physics subjects (high school, college, conceptual, astronomy)
  2. GPQA Diamond physics subset
  3. Custom curated HEP and graduate-level problems

Usage:
    python -m training.data.build_physics_problems \
        --benchmarks-dir /work/hdd/bgde/jhill5/data/benchmarks \
        --output /work/hdd/bgde/jhill5/data/physics_problems.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Graduate-level physics problems for trace generation.
# These cover areas where GRACE needs strong reasoning:
# quantum mechanics, particle physics, statistical mechanics,
# electrodynamics, general relativity, nuclear physics.
CURATED_PROBLEMS = [
    # Quantum Mechanics
    {
        "problem": "A spin-1/2 particle is in the state |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5. What is the probability of measuring spin-up along the x-axis? What is the expectation value of S_z?",
        "domain": "quantum_mechanics",
        "difficulty": "medium",
    },
    {
        "problem": "Derive the energy eigenvalues for a particle in a finite square well potential of depth V₀ and width 2a. Under what condition does at least one bound state exist?",
        "domain": "quantum_mechanics",
        "difficulty": "hard",
    },
    {
        "problem": "Using time-dependent perturbation theory, calculate the transition rate for a hydrogen atom in the ground state to transition to the 2p state when exposed to a linearly polarized electromagnetic wave. Express your answer in terms of the electric field amplitude and relevant atomic quantities.",
        "domain": "quantum_mechanics",
        "difficulty": "hard",
    },
    {
        "problem": "A quantum harmonic oscillator is in the coherent state |α⟩ with α = 2 + i. Calculate the expectation values ⟨x⟩, ⟨p⟩, ⟨x²⟩, ⟨p²⟩, and verify the uncertainty relation.",
        "domain": "quantum_mechanics",
        "difficulty": "medium",
    },
    # Particle Physics / HEP
    {
        "problem": "Calculate the differential cross-section dσ/dΩ for electron-muon scattering (e⁻μ⁻ → e⁻μ⁻) in the center-of-mass frame at tree level in QED. Express your result in terms of the Mandelstam variables.",
        "domain": "particle_physics",
        "difficulty": "hard",
    },
    {
        "problem": "The Higgs boson decays to two photons via a loop diagram. Explain why this process cannot occur at tree level in the Standard Model. What particles contribute to the dominant loop diagrams, and why is the top quark contribution the largest?",
        "domain": "particle_physics",
        "difficulty": "hard",
    },
    {
        "problem": "In a proton-proton collision at √s = 13 TeV, estimate the production cross-section for tt̄ pair production at leading order. What is the dominant production mechanism at this energy?",
        "domain": "particle_physics",
        "difficulty": "hard",
    },
    {
        "problem": "Derive the Breit-Wigner resonance formula for a particle of mass M and total decay width Γ. Apply it to estimate the cross-section for e⁺e⁻ → Z → μ⁺μ⁻ near the Z pole.",
        "domain": "particle_physics",
        "difficulty": "hard",
    },
    {
        "problem": "Explain the role of the CKM matrix in weak decays. Why does the small value of V_ub make B meson decays to charmless final states rare? Estimate the branching ratio B(B⁺ → π⁺π⁰) relative to B(B⁺ → D̄⁰π⁺).",
        "domain": "particle_physics",
        "difficulty": "hard",
    },
    {
        "problem": "A particle detector measures the invariant mass of two muons from proton-proton collisions. The distribution shows a peak near 91 GeV and another near 3.1 GeV. Identify the resonances and explain how to extract the signal from the combinatorial background.",
        "domain": "particle_physics",
        "difficulty": "medium",
    },
    # Statistical Mechanics
    {
        "problem": "Derive the Bose-Einstein distribution function from the grand canonical ensemble. At what temperature does an ideal Bose gas undergo Bose-Einstein condensation? Express T_c in terms of the particle density and mass.",
        "domain": "statistical_mechanics",
        "difficulty": "hard",
    },
    {
        "problem": "Using the Ising model in 1D with N spins and nearest-neighbor interaction J, calculate the partition function exactly using the transfer matrix method. Show that there is no phase transition at finite temperature.",
        "domain": "statistical_mechanics",
        "difficulty": "hard",
    },
    {
        "problem": "A system of N non-interacting fermions is confined to a 3D box of volume V at temperature T. Derive the Fermi energy and show that the specific heat at low temperatures is proportional to T.",
        "domain": "statistical_mechanics",
        "difficulty": "medium",
    },
    # Electrodynamics
    {
        "problem": "A point charge q moves with constant velocity v along the x-axis. Using the Liénard-Wiechert potentials, derive the electric and magnetic fields at an arbitrary field point. Show that the fields reduce to the Coulomb field when v → 0.",
        "domain": "electrodynamics",
        "difficulty": "hard",
    },
    {
        "problem": "Derive the Fresnel equations for reflection and transmission of an electromagnetic wave at a planar interface between two dielectrics. At what angle of incidence is the reflected wave completely polarized?",
        "domain": "electrodynamics",
        "difficulty": "medium",
    },
    {
        "problem": "A conducting sphere of radius a is placed in a uniform external electric field E₀. Solve for the potential everywhere using the method of separation of variables in spherical coordinates. Find the induced surface charge density.",
        "domain": "electrodynamics",
        "difficulty": "medium",
    },
    # General Relativity
    {
        "problem": "Starting from the Schwarzschild metric, derive the equation for the deflection of light passing near a massive body of mass M. Show that the deflection angle is 4GM/(c²b) where b is the impact parameter.",
        "domain": "general_relativity",
        "difficulty": "hard",
    },
    {
        "problem": "Derive the gravitational redshift formula from the equivalence principle. A photon is emitted at the surface of a neutron star with mass 1.4 M_☉ and radius 10 km. What is the fractional change in wavelength observed by a distant observer?",
        "domain": "general_relativity",
        "difficulty": "medium",
    },
    # Nuclear Physics
    {
        "problem": "Using the semi-empirical mass formula (Bethe-Weizsäcker formula), calculate the binding energy of Fe-56 and compare it to the experimental value. Which term contributes most to the binding energy per nucleon at A=56?",
        "domain": "nuclear_physics",
        "difficulty": "medium",
    },
    {
        "problem": "Explain the physics of beta decay in terms of the weak interaction. Why is the neutrino spectrum continuous in β⁻ decay? Calculate the Q-value for the decay of a free neutron.",
        "domain": "nuclear_physics",
        "difficulty": "medium",
    },
    # Condensed Matter
    {
        "problem": "Derive the Bloch theorem for electrons in a periodic potential. Starting from the nearly-free electron model, show how energy gaps appear at the Brillouin zone boundaries and estimate the gap size for a weak periodic potential V(x) = 2V₁cos(2πx/a).",
        "domain": "condensed_matter",
        "difficulty": "hard",
    },
    {
        "problem": "In BCS theory, explain how Cooper pairs form and why the superconducting gap Δ is related to the critical temperature by Δ(0) ≈ 1.76 k_B T_c. Estimate T_c for a material with a Debye temperature of 300 K and electron-phonon coupling constant λ = 0.3.",
        "domain": "condensed_matter",
        "difficulty": "hard",
    },
    # Astrophysics
    {
        "problem": "Derive the Chandrasekhar mass limit for white dwarfs using dimensional analysis and the balance between gravitational pressure and electron degeneracy pressure. Why does this limit not apply to neutron stars in the same way?",
        "domain": "astrophysics",
        "difficulty": "hard",
    },
    {
        "problem": "A Type Ia supernova has peak absolute magnitude M = -19.3. If it is observed with apparent magnitude m = 15.2, what is its distance? Account for the effects of cosmological redshift if z = 0.03.",
        "domain": "astrophysics",
        "difficulty": "medium",
    },
    # Detector Physics (directly relevant to GRACE)
    {
        "problem": "A liquid argon TPC detects ionization electrons and scintillation photons. If the scintillation yield is 40,000 photons/MeV and the photon detection efficiency is 5%, what is the expected photoelectron count for a 1 MeV energy deposit? What determines the energy resolution?",
        "domain": "detector_physics",
        "difficulty": "medium",
    },
    {
        "problem": "Design the electromagnetic calorimeter section of a general-purpose particle detector for a 100 TeV proton-proton collider. Discuss the choice of absorber material, sampling fraction, and the resulting energy resolution σ(E)/E. How does the Molière radius affect the lateral segmentation?",
        "domain": "detector_physics",
        "difficulty": "hard",
    },
    {
        "problem": "Explain how a silicon strip detector measures the position of a charged particle. If the strip pitch is 50 μm and charge sharing is used, what position resolution can be achieved? How does radiation damage affect the detector performance over time?",
        "domain": "detector_physics",
        "difficulty": "medium",
    },
    {
        "problem": "In a sampling calorimeter with lead absorber plates and plastic scintillator active layers, derive the energy resolution as a function of the sampling fraction. A 10 GeV electron shower produces 500 MeV in the scintillator. What is the stochastic term of the resolution?",
        "domain": "detector_physics",
        "difficulty": "hard",
    },
]


def load_mmlu_physics(benchmarks_dir: Path) -> list[dict]:
    """Load MMLU physics questions and format as problems."""
    mmlu_path = benchmarks_dir / "mmlu_stem" / "mmlu_stem.jsonl"
    if not mmlu_path.exists():
        print(f"  WARNING: MMLU data not found at {mmlu_path}")
        return []

    problems = []
    physics_domains = {"physics"}
    with open(mmlu_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("domain") == "physics":
                problems.append({
                    "id": f"mmlu_{len(problems)}",
                    "problem": row["query"],
                    "domain": row.get("subject", "physics"),
                    "difficulty": "medium",
                    "source": "mmlu",
                })

    return problems


def load_grace_hep(grace_jsonl: Path) -> list[dict]:
    """Load HEP physics problems extracted from GRACE.

    Expects the JSONL produced by `training.data.extract_hep_physics`. Each
    record has {problem, domain, difficulty, source}; we add an `id` field
    and forward `source` as-is so trace filtering can stratify by GRACE
    sub-source (scintillator_kg, detector_geometry_kg, etc.).
    """
    if not grace_jsonl.exists():
        print(f"  WARNING: GRACE HEP data not found at {grace_jsonl}")
        return []

    problems: list[dict] = []
    with open(grace_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problems.append({
                "id": f"grace_hep_{len(problems)}",
                "problem": row["problem"],
                "domain": row.get("domain", "hep_experiment"),
                "difficulty": row.get("difficulty", "hard"),
                "source": row.get("source", "grace/hep_physics"),
            })
    return problems


def load_gpqa_physics(benchmarks_dir: Path) -> list[dict]:
    """Load GPQA Diamond questions (all are graduate-level science)."""
    gpqa_path = benchmarks_dir / "gpqa" / "gpqa_diamond.jsonl"
    if not gpqa_path.exists():
        print(f"  WARNING: GPQA data not found at {gpqa_path}")
        return []

    problems = []
    with open(gpqa_path) as f:
        for line in f:
            row = json.loads(line)
            # Format as open-ended problem (not multiple choice)
            # Include the choices as context but ask for reasoning
            problem_text = row["question"]
            choices = f"\n(A) {row['choice_a']}\n(B) {row['choice_b']}\n(C) {row['choice_c']}\n(D) {row['choice_d']}"

            problems.append({
                "id": f"gpqa_{len(problems)}",
                "problem": f"{problem_text}\n{choices}\n\nExplain your reasoning step by step, then give your answer.",
                "answer": row["correct"],
                "domain": row.get("domain", "physics"),
                "difficulty": "hard",
                "source": "gpqa",
            })

    return problems


def main():
    parser = argparse.ArgumentParser(description="Build physics problem set for trace generation")
    parser.add_argument(
        "--benchmarks-dir",
        default="/work/hdd/bgde/jhill5/data/benchmarks",
        help="Directory containing downloaded benchmarks",
    )
    parser.add_argument(
        "--output",
        default="/work/hdd/bgde/jhill5/data/physics_problems.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--grace-hep-jsonl",
        default="/work/hdd/bgde/jhill5/data/hep_physics_problems.jsonl",
        help="Path to GRACE-extracted HEP physics problems JSONL "
             "(produced by training.data.extract_hep_physics).",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    all_problems = []

    # 1. Curated graduate-level physics problems
    print("[problems] adding curated physics problems...")
    for i, p in enumerate(CURATED_PROBLEMS):
        p["id"] = f"curated_{i}"
        p["source"] = "curated"
        all_problems.append(p)
    print(f"  -> {len(CURATED_PROBLEMS)} curated problems")

    # 2. MMLU physics
    print("[problems] loading MMLU physics...")
    mmlu = load_mmlu_physics(benchmarks_dir)
    all_problems.extend(mmlu)
    print(f"  -> {len(mmlu)} MMLU physics problems")

    # 3. GPQA Diamond
    print("[problems] loading GPQA Diamond...")
    gpqa = load_gpqa_physics(benchmarks_dir)
    all_problems.extend(gpqa)
    print(f"  -> {len(gpqa)} GPQA problems")

    # 4. GRACE HEP physics extractions
    print("[problems] loading GRACE HEP physics...")
    grace_hep = load_grace_hep(Path(args.grace_hep_jsonl))
    all_problems.extend(grace_hep)
    print(f"  -> {len(grace_hep)} GRACE HEP problems")

    # Assign IDs
    for i, p in enumerate(all_problems):
        if "id" not in p:
            p["id"] = f"prob_{i}"

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    sources = Counter(p.get("source", "unknown") for p in all_problems)
    print(f"\n[problems] total: {len(all_problems)} physics problems")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"  saved to: {output_path}")


if __name__ == "__main__":
    main()
