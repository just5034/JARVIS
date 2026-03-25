"""Download all benchmark datasets for JARVIS evaluation and training.

Downloads:
  - GPQA Diamond (198 physics/science MCQ) → gpqa/
  - AIME 2024 (30 competition math problems) → aime/
  - LiveCodeBench (code generation problems) → livecode/
  - MMLU-STEM subset (for router training data) → mmlu_stem/
  - Generates router eval data → router/

Usage:
    # On Delta (download to scratch):
    python -m training.data.download_benchmarks \
        --output /scratch/bgde/jhill5/data/benchmarks

    # Locally (for development):
    python -m training.data.download_benchmarks --output ./data/benchmarks
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def download_gpqa(output_dir: Path) -> int:
    """Download GPQA Diamond from HuggingFace."""
    from datasets import load_dataset

    out = output_dir / "gpqa"
    out.mkdir(parents=True, exist_ok=True)

    print("[download] GPQA Diamond...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    problems = []
    for row in ds:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        indices = list(range(4))
        random.seed(hash(row["Question"]))
        random.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_idx = indices.index(0)
        correct_letter = "ABCD"[correct_idx]

        problems.append({
            "question": row["Question"],
            "choice_a": shuffled[0],
            "choice_b": shuffled[1],
            "choice_c": shuffled[2],
            "choice_d": shuffled[3],
            "correct": correct_letter,
            "domain": row.get("Subdomain", "unknown"),
        })

    with open(out / "gpqa_diamond.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"  → {len(problems)} problems saved to {out / 'gpqa_diamond.jsonl'}")
    return len(problems)


def download_aime(output_dir: Path) -> int:
    """Download AIME 2024 problems from HuggingFace."""
    from datasets import load_dataset

    out = output_dir / "aime"
    out.mkdir(parents=True, exist_ok=True)

    print("[download] AIME 2024...")

    # Try multiple dataset sources
    ds = None
    for source in ["Maxwell-Jia/AIME_2024", "qq8933/AIME_2024"]:
        try:
            ds = load_dataset(source, split="train")
            print(f"  loaded from {source}")
            break
        except Exception:
            try:
                ds = load_dataset(source, split="test")
                print(f"  loaded from {source} (test split)")
                break
            except Exception:
                continue

    if ds is None:
        print("  WARNING: could not download AIME 2024 from HuggingFace")
        print("  You can manually create data/benchmarks/aime/aime_2024.jsonl")
        print("  Format: {\"problem\": str, \"answer\": str} per line")
        return 0

    problems = []
    for row in ds:
        problem_text = row.get("problem", row.get("Problem", row.get("question", "")))
        answer = str(row.get("answer", row.get("Answer", row.get("solution", ""))))
        if problem_text and answer:
            problems.append({
                "problem": problem_text,
                "answer": answer.strip(),
                "contest": row.get("contest", "AIME 2024"),
                "number": row.get("number", row.get("problem_number", len(problems) + 1)),
            })

    with open(out / "aime_2024.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"  → {len(problems)} problems saved to {out / 'aime_2024.jsonl'}")
    return len(problems)


def download_livecode(output_dir: Path) -> int:
    """Download LiveCodeBench from HuggingFace."""
    from datasets import load_dataset

    out = output_dir / "livecode"
    out.mkdir(parents=True, exist_ok=True)

    print("[download] LiveCodeBench...")
    try:
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
    except Exception:
        ds = load_dataset("livecodebench/code_generation_lite", split="train")

    problems = []
    for row in ds:
        problem = {
            "id": row.get("question_id", row.get("id", len(problems))),
            "title": row.get("question_title", row.get("title", "")),
            "description": row.get("question_content", row.get("description", "")),
            "difficulty": row.get("difficulty", "unknown"),
            "input_format": row.get("input_format", ""),
            "output_format": row.get("output_format", ""),
            "constraints": row.get("constraints", ""),
        }

        if "public_test_cases" in row:
            test_data = row["public_test_cases"]
            if isinstance(test_data, str):
                try:
                    test_data = json.loads(test_data)
                except json.JSONDecodeError:
                    test_data = []
            problem["test_cases"] = test_data
        elif "test_cases" in row:
            problem["test_cases"] = row["test_cases"]
        else:
            problem["test_cases"] = []

        problems.append(problem)

    with open(out / "livecode_bench.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"  → {len(problems)} problems saved to {out / 'livecode_bench.jsonl'}")
    return len(problems)


def download_mmlu_stem(output_dir: Path) -> int:
    """Download MMLU STEM subjects for router training data."""
    from datasets import load_dataset

    out = output_dir / "mmlu_stem"
    out.mkdir(parents=True, exist_ok=True)

    print("[download] MMLU STEM subset (for router training)...")

    # STEM subjects relevant to JARVIS domains
    subjects = {
        "physics": [
            "high_school_physics",
            "college_physics",
            "conceptual_physics",
            "astronomy",
        ],
        "math": [
            "high_school_mathematics",
            "college_mathematics",
            "abstract_algebra",
            "high_school_statistics",
        ],
        "chemistry": [
            "high_school_chemistry",
            "college_chemistry",
        ],
        "biology": [
            "high_school_biology",
            "college_biology",
            "anatomy",
        ],
        "code": [
            "high_school_computer_science",
            "college_computer_science",
            "machine_learning",
        ],
    }

    total = 0
    all_examples = []

    for domain, subject_list in subjects.items():
        for subject in subject_list:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                for row in ds:
                    all_examples.append({
                        "query": row["question"],
                        "domain": domain,
                        "subject": subject,
                        "difficulty": "medium",  # MMLU is roughly medium
                    })
                total += len(ds)
            except Exception as e:
                print(f"  WARNING: could not load {subject}: {e}")

    with open(out / "mmlu_stem.jsonl", "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  → {total} examples across {len(subjects)} domains saved")
    return total


def generate_router_eval(output_dir: Path) -> int:
    """Generate router evaluation data from multiple sources."""
    out = output_dir / "router"
    out.mkdir(parents=True, exist_ok=True)

    print("[generate] Router evaluation data...")

    eval_data = []

    # HEP-specific queries (for HEP detection eval)
    hep_queries = [
        {"query": "Calculate the Higgs boson decay width to two photons", "domain": "physics", "difficulty": "hard", "is_hep": True},
        {"query": "What is the parton distribution function of the gluon at x=0.01?", "domain": "physics", "difficulty": "hard", "is_hep": True},
        {"query": "Simulate proton-proton collisions at 13 TeV using Pythia8", "domain": "code", "difficulty": "hard", "is_hep": True},
        {"query": "Write a Geant4 detector geometry for an electromagnetic calorimeter", "domain": "code", "difficulty": "hard", "is_hep": True},
        {"query": "Compute the cross-section for ttbar production at the LHC", "domain": "physics", "difficulty": "hard", "is_hep": True},
        {"query": "Analyze the ATLAS dimuon invariant mass spectrum for Z boson resonance", "domain": "physics", "difficulty": "hard", "is_hep": True},
        {"query": "Implement jet clustering using the anti-kt algorithm with FastJet", "domain": "code", "difficulty": "medium", "is_hep": True},
        {"query": "Apply Delphes fast simulation to reconstruct particle-level objects", "domain": "code", "difficulty": "medium", "is_hep": True},
        {"query": "Calculate the branching ratio of B meson to K*mu+mu-", "domain": "physics", "difficulty": "hard", "is_hep": True},
        {"query": "What is the CMS trigger efficiency for single lepton events?", "domain": "physics", "difficulty": "medium", "is_hep": True},
    ]
    eval_data.extend(hep_queries)

    # Non-HEP physics
    non_hep_physics = [
        {"query": "Derive the Schwarzschild metric from Einstein field equations", "domain": "physics", "difficulty": "hard", "is_hep": False},
        {"query": "What is the refractive index of diamond?", "domain": "physics", "difficulty": "easy", "is_hep": False},
        {"query": "Solve the quantum harmonic oscillator using ladder operators", "domain": "physics", "difficulty": "medium", "is_hep": False},
        {"query": "Calculate the magnetic field inside a solenoid", "domain": "physics", "difficulty": "easy", "is_hep": False},
        {"query": "What is the Chandrasekhar mass limit for white dwarfs?", "domain": "physics", "difficulty": "medium", "is_hep": False},
    ]
    eval_data.extend(non_hep_physics)

    # Math queries
    math_queries = [
        {"query": "Find all integer solutions to x^3 + y^3 = z^3 + w^3 where x+y=z+w", "domain": "math", "difficulty": "hard", "is_hep": False},
        {"query": "Prove that the sum of reciprocals of primes diverges", "domain": "math", "difficulty": "hard", "is_hep": False},
        {"query": "Compute the determinant of a 4x4 matrix", "domain": "math", "difficulty": "easy", "is_hep": False},
        {"query": "What is the integral of e^(-x^2) from 0 to infinity?", "domain": "math", "difficulty": "medium", "is_hep": False},
        {"query": "Find the eigenvalues of [[3,1],[1,3]]", "domain": "math", "difficulty": "easy", "is_hep": False},
    ]
    eval_data.extend(math_queries)

    # Code queries
    code_queries = [
        {"query": "Write a Python function to find the longest common subsequence", "domain": "code", "difficulty": "medium", "is_hep": False},
        {"query": "Implement a red-black tree in C++", "domain": "code", "difficulty": "hard", "is_hep": False},
        {"query": "Write a SQL query to find the second highest salary", "domain": "code", "difficulty": "easy", "is_hep": False},
        {"query": "Implement gradient descent for linear regression from scratch", "domain": "code", "difficulty": "medium", "is_hep": False},
        {"query": "Build a concurrent web scraper using asyncio in Python", "domain": "code", "difficulty": "medium", "is_hep": False},
    ]
    eval_data.extend(code_queries)

    # Chemistry queries
    chem_queries = [
        {"query": "What is the VSEPR geometry of SF6?", "domain": "chemistry", "difficulty": "easy", "is_hep": False},
        {"query": "Design a retrosynthetic route for ibuprofen", "domain": "chemistry", "difficulty": "hard", "is_hep": False},
        {"query": "Calculate the pH of a 0.1M acetic acid solution", "domain": "chemistry", "difficulty": "medium", "is_hep": False},
    ]
    eval_data.extend(chem_queries)

    # Biology queries
    bio_queries = [
        {"query": "Explain the mechanism of CRISPR-Cas9 gene editing", "domain": "biology", "difficulty": "medium", "is_hep": False},
        {"query": "What are the steps of the citric acid cycle?", "domain": "biology", "difficulty": "medium", "is_hep": False},
        {"query": "Describe the role of p53 in cell cycle regulation", "domain": "biology", "difficulty": "medium", "is_hep": False},
    ]
    eval_data.extend(bio_queries)

    # Protein queries
    protein_queries = [
        {"query": "Predict the secondary structure of the sequence MKWVTFISLLFLFSSAYS", "domain": "protein", "difficulty": "medium", "is_hep": False},
        {"query": "What is the function of hemoglobin's quaternary structure?", "domain": "protein", "difficulty": "easy", "is_hep": False},
    ]
    eval_data.extend(protein_queries)

    # Genomics queries
    genomics_queries = [
        {"query": "Identify potential open reading frames in this DNA sequence: ATGATCGATCGATCGATGA", "domain": "genomics", "difficulty": "medium", "is_hep": False},
        {"query": "What is the GC content of the E. coli genome?", "domain": "genomics", "difficulty": "easy", "is_hep": False},
    ]
    eval_data.extend(genomics_queries)

    # General queries
    general_queries = [
        {"query": "What is the capital of France?", "domain": "general", "difficulty": "easy", "is_hep": False},
        {"query": "Summarize the key arguments in Rawls' Theory of Justice", "domain": "general", "difficulty": "medium", "is_hep": False},
        {"query": "What are the main causes of the 2008 financial crisis?", "domain": "general", "difficulty": "medium", "is_hep": False},
    ]
    eval_data.extend(general_queries)

    with open(out / "router_eval.jsonl", "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    print(f"  → {len(eval_data)} eval examples saved to {out / 'router_eval.jsonl'}")
    return len(eval_data)


def main():
    parser = argparse.ArgumentParser(description="Download all JARVIS benchmark datasets")
    parser.add_argument(
        "--output",
        default="/scratch/bgde/jhill5/data/benchmarks",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
        help="Skip MMLU download (large, only needed for router training)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JARVIS Benchmark Data Download")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    totals = {}

    try:
        totals["gpqa"] = download_gpqa(output_dir)
    except Exception as e:
        print(f"  ERROR downloading GPQA: {e}")
        totals["gpqa"] = 0

    try:
        totals["aime"] = download_aime(output_dir)
    except Exception as e:
        print(f"  ERROR downloading AIME: {e}")
        totals["aime"] = 0

    try:
        totals["livecode"] = download_livecode(output_dir)
    except Exception as e:
        print(f"  ERROR downloading LiveCodeBench: {e}")
        totals["livecode"] = 0

    if not args.skip_mmlu:
        try:
            totals["mmlu_stem"] = download_mmlu_stem(output_dir)
        except Exception as e:
            print(f"  ERROR downloading MMLU: {e}")
            totals["mmlu_stem"] = 0

    totals["router_eval"] = generate_router_eval(output_dir)

    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, count in totals.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {name:20s} {count:6d} examples  [{status}]")

    # Save manifest
    manifest = {"datasets": totals, "output_dir": str(output_dir)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
