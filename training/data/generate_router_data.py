"""Generate training data for the JARVIS router classifier.

Sources:
  1. MMLU STEM questions (labeled by subject → domain)
  2. Synthetic domain-specific queries via templates
  3. HEP-specific queries for subdomain detection
  4. Difficulty labels derived from model performance (Phase 4H)

This script generates the INITIAL training set from templates + MMLU.
After brains are trained, run with --add-difficulty to label difficulty
from actual model performance.

Usage:
    # Generate initial training set (domain labels only):
    python -m training.data.generate_router_data \
        --output /scratch/bgde-delta-gpu/data/router_training.jsonl \
        --mmlu-dir /scratch/bgde-delta-gpu/data/benchmarks/mmlu_stem

    # Add difficulty labels after brains are trained (Phase 4H):
    python -m training.data.generate_router_data \
        --output /scratch/bgde-delta-gpu/data/router_training_with_difficulty.jsonl \
        --add-difficulty \
        --model /projects/bgde-delta-gpu/models/r1-distill-qwen-32b \
        --input /scratch/bgde-delta-gpu/data/router_training.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Domain-specific query templates for synthetic data generation.
# Each template produces queries that clearly belong to one domain.
TEMPLATES = {
    "physics": [
        "Calculate the {quantity} of a {system} with {parameter}",
        "Derive the equation of motion for {system}",
        "What is the {quantity} in {context}?",
        "Solve the {equation_type} equation for a {system}",
        "Explain the {phenomenon} in {context}",
    ],
    "math": [
        "Prove that {statement}",
        "Find all {objects} satisfying {condition}",
        "Compute the {operation} of {expression}",
        "Let {setup}. Show that {conclusion}",
        "Evaluate the {math_type} {expression}",
    ],
    "code": [
        "Write a {language} function to {task}",
        "Implement {algorithm} in {language}",
        "Debug the following {language} code: {snippet}",
        "Optimize this {language} function for {metric}",
        "Write unit tests for {function_desc}",
    ],
    "chemistry": [
        "What is the {property} of {compound}?",
        "Draw the Lewis structure of {molecule}",
        "Calculate the {quantity} for the reaction {reaction}",
        "Predict the products of {reaction_type} between {reactants}",
        "Explain the mechanism of {reaction_type}",
    ],
    "biology": [
        "Describe the process of {biological_process}",
        "What is the role of {molecule} in {pathway}?",
        "Explain how {organism} adapts to {environment}",
        "Compare {structure_a} and {structure_b} in {context}",
        "What are the stages of {biological_process}?",
    ],
    "protein": [
        "Predict the structure of the protein sequence {sequence}",
        "What is the binding affinity of {protein} to {ligand}?",
        "Identify the active site residues of {enzyme}",
        "Model the folding pathway of {protein}",
        "Analyze the protein-protein interaction between {protein_a} and {protein_b}",
    ],
    "genomics": [
        "Annotate the following DNA sequence: {sequence}",
        "Identify variants in the {gene} region from this VCF data",
        "What is the function of {gene} in {organism}?",
        "Design CRISPR guide RNAs targeting {gene}",
        "Analyze the RNA-seq expression profile of {gene_set}",
    ],
    "general": [
        "What is {topic}?",
        "Explain {concept} in simple terms",
        "Summarize the key points of {subject}",
        "Compare {option_a} and {option_b}",
        "What are the implications of {event}?",
    ],
}

# Slot fillers for template instantiation
FILLERS = {
    "physics": {
        "quantity": ["energy", "momentum", "angular momentum", "entropy", "wavelength",
                     "frequency", "impedance", "conductivity", "permittivity", "potential"],
        "system": ["harmonic oscillator", "rigid body", "ideal gas", "black body",
                   "charged particle", "superconductor", "neutron star", "Bose-Einstein condensate"],
        "parameter": ["mass m=5kg", "temperature T=300K", "charge q=1.6e-19 C",
                      "velocity v=0.9c", "magnetic field B=1T"],
        "context": ["quantum mechanics", "special relativity", "thermodynamics",
                    "electromagnetism", "condensed matter physics", "astrophysics"],
        "equation_type": ["Schrodinger", "wave", "heat", "Maxwell", "Navier-Stokes"],
        "phenomenon": ["superconductivity", "superfluidity", "photoelectric effect",
                       "Compton scattering", "Zeeman effect", "Doppler effect"],
    },
    "math": {
        "statement": ["sqrt(2) is irrational", "there are infinitely many primes",
                      "every continuous function on [0,1] is bounded",
                      "the fundamental theorem of algebra holds"],
        "objects": ["integers", "prime numbers", "real roots", "eigenvalues",
                    "fixed points", "subgroups"],
        "condition": ["x^2+y^2=z^2", "f(f(x))=x", "det(A)=0", "gcd(a,b)=1"],
        "operation": ["integral", "derivative", "limit", "determinant",
                      "trace", "rank", "nullity"],
        "expression": ["sin(x)/x as x→0", "sum_{n=1}^{inf} 1/n^2",
                       "integral of ln(x) dx", "d/dx[x^x]"],
        "setup": ["f be a continuous function on [a,b]",
                  "G be a finite group of order n",
                  "V be an n-dimensional vector space"],
        "conclusion": ["f attains its maximum", "G has a subgroup of order p",
                       "dim(V)+dim(V^perp)=n"],
        "math_type": ["definite integral", "infinite series", "limit",
                      "double integral", "contour integral"],
    },
    "code": {
        "language": ["Python", "C++", "Java", "Rust", "Go", "JavaScript"],
        "task": ["sort a linked list", "find shortest path in a graph",
                 "parse a JSON file", "implement a thread pool",
                 "build a REST API client", "compress a file using Huffman coding"],
        "algorithm": ["quicksort", "Dijkstra's algorithm", "A* search",
                      "dynamic programming for knapsack", "binary search tree",
                      "topological sort"],
        "snippet": ["def foo(x): return x + 1", "for i in range(n): ...",
                    "class Node: ..."],
        "metric": ["time complexity", "memory usage", "cache performance"],
        "function_desc": ["a binary search implementation",
                          "a database connection pool",
                          "a rate limiter"],
    },
    "chemistry": {
        "property": ["boiling point", "electronegativity", "oxidation state",
                     "bond length", "enthalpy of formation", "solubility"],
        "compound": ["NaCl", "H2SO4", "benzene", "ethanol", "glucose",
                     "aspirin", "caffeine"],
        "molecule": ["CO2", "H2O", "NH3", "CH4", "O3", "NO2"],
        "quantity": ["Gibbs free energy", "equilibrium constant", "pH",
                     "cell potential", "enthalpy change"],
        "reaction": ["combustion of methane", "neutralization of HCl with NaOH",
                     "electrolysis of water"],
        "reaction_type": ["SN1 substitution", "Diels-Alder",
                          "aldol condensation", "Friedel-Crafts alkylation"],
        "reactants": ["Na and Cl2", "Fe and O2", "CH4 and O2"],
    },
    "biology": {
        "biological_process": ["mitosis", "meiosis", "photosynthesis",
                               "cellular respiration", "DNA replication",
                               "transcription", "translation"],
        "molecule": ["insulin", "hemoglobin", "ATP", "DNA polymerase",
                     "RNA polymerase", "tubulin"],
        "pathway": ["glycolysis", "citric acid cycle", "electron transport chain",
                    "Calvin cycle", "signal transduction"],
        "organism": ["E. coli", "Drosophila", "Arabidopsis", "zebrafish"],
        "environment": ["high salinity", "low oxygen", "extreme heat"],
        "structure_a": ["mitochondria", "prokaryotic cells", "arteries"],
        "structure_b": ["chloroplasts", "eukaryotic cells", "veins"],
        "context": ["eukaryotes", "prokaryotes", "mammals", "plants"],
    },
    "protein": {
        "sequence": ["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
                     "MKWVTFISLLFLFSSAYS", "MNIFEMLRIDEGLRLKIYKDTEG"],
        "protein": ["p53", "EGFR", "insulin receptor", "lysozyme", "GFP"],
        "ligand": ["ATP", "glucose", "inhibitor X", "substrate Y"],
        "enzyme": ["trypsin", "HIV protease", "carbonic anhydrase"],
        "protein_a": ["actin", "CDK2", "Ras"],
        "protein_b": ["myosin", "cyclin A", "Raf"],
    },
    "genomics": {
        "sequence": ["ATGATCGATCGATCGATGA", "GCTAGCTAGCTAGCTAG",
                     "TTAACCGGTTAACCGG"],
        "gene": ["BRCA1", "TP53", "EGFR", "KRAS", "MYC"],
        "organism": ["human", "mouse", "yeast", "Arabidopsis"],
        "gene_set": ["{BRCA1, TP53, RB1}", "{MYC, FOS, JUN}",
                     "{SOX2, OCT4, NANOG}"],
    },
    "general": {
        "topic": ["climate change", "artificial intelligence",
                  "the theory of relativity", "blockchain technology"],
        "concept": ["machine learning", "supply chain management",
                    "behavioral economics", "game theory"],
        "subject": ["the French Revolution", "quantum computing",
                    "sustainable development", "CRISPR technology"],
        "option_a": ["solar energy", "remote work", "Python"],
        "option_b": ["wind energy", "office work", "JavaScript"],
        "event": ["the discovery of gravitational waves",
                  "the development of mRNA vaccines",
                  "the rise of large language models"],
    },
}

# HEP-specific queries (always labeled as HEP for subdomain detection)
HEP_QUERIES = [
    "Simulate Higgs boson production via gluon fusion at 14 TeV",
    "Calculate the NLO QCD corrections to top quark pair production",
    "What is the expected number of signal events for H→γγ at the LHC?",
    "Write a ROOT macro to fit the dimuon invariant mass spectrum",
    "Configure Pythia8 to generate minimum bias events at 13.6 TeV",
    "Implement a Delphes card for the CMS Phase-2 detector upgrade",
    "Analyze the missing transverse energy distribution in SUSY searches",
    "Calculate the b-tagging efficiency for jets with pT > 30 GeV",
    "What is the acceptance times efficiency for the Z→ee channel?",
    "Derive the sensitivity reach for a dark photon search at Belle II",
    "Write a Geant4 simulation for a sampling calorimeter with lead absorbers",
    "Compute the parton-level cross section for qq→Z using MCFM",
    "Reconstruct the W boson mass from leptonic decay products",
    "Implement a BDT classifier for signal/background separation in ttH analysis",
    "Calculate the systematic uncertainty from jet energy scale variations",
    "Design a trigger menu for selecting VBF Higgs events",
    "What are the dominant backgrounds to the H→WW→lνlν search?",
    "Write a FastJet plugin for jet substructure analysis",
    "Compute the luminosity uncertainty for the 2024 LHC Run 3 data",
    "Simulate optical photon propagation in a liquid argon TPC using Geant4",
]


def instantiate_templates(domain: str, n_per_template: int = 20) -> list[dict]:
    """Generate synthetic queries by filling templates with random slot values."""
    templates = TEMPLATES[domain]
    fillers = FILLERS.get(domain, {})
    results = []

    for template in templates:
        for _ in range(n_per_template):
            query = template
            for slot, values in fillers.items():
                placeholder = "{" + slot + "}"
                if placeholder in query:
                    query = query.replace(placeholder, random.choice(values), 1)

            # Only add if all placeholders were filled
            if "{" not in query:
                results.append({
                    "query": query,
                    "domain": domain,
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "is_hep": False,
                    "source": "synthetic",
                })

    return results


def load_mmlu_data(mmlu_dir: str) -> list[dict]:
    """Load MMLU STEM data as router training examples."""
    path = Path(mmlu_dir) / "mmlu_stem.jsonl"
    if not path.exists():
        print(f"  MMLU data not found at {path} — skipping")
        return []

    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append({
                "query": row["query"],
                "domain": row["domain"],
                "difficulty": row.get("difficulty", "medium"),
                "is_hep": False,
                "source": "mmlu",
            })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate router training data")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--mmlu-dir",
        default="/scratch/bgde-delta-gpu/data/benchmarks/mmlu_stem",
        help="Path to downloaded MMLU STEM data",
    )
    parser.add_argument(
        "--n-per-template",
        type=int,
        default=20,
        help="Synthetic examples per template",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    all_data = []

    # 1. Synthetic template-based queries
    print("[router-data] generating synthetic queries...")
    for domain in TEMPLATES:
        examples = instantiate_templates(domain, n_per_template=args.n_per_template)
        all_data.extend(examples)
        print(f"  {domain}: {len(examples)} synthetic queries")

    # 2. HEP-specific queries
    print("[router-data] adding HEP queries...")
    for query in HEP_QUERIES:
        # Classify HEP queries as physics or code based on content
        code_indicators = ["write", "implement", "simulate", "configure", "design", "macro"]
        domain = "code" if any(w in query.lower() for w in code_indicators) else "physics"
        all_data.append({
            "query": query,
            "domain": domain,
            "difficulty": "hard",
            "is_hep": True,
            "source": "hep_curated",
        })
    print(f"  HEP: {len(HEP_QUERIES)} curated queries")

    # 3. MMLU STEM data
    mmlu_data = load_mmlu_data(args.mmlu_dir)
    if mmlu_data:
        all_data.extend(mmlu_data)
        print(f"  MMLU: {len(mmlu_data)} examples")

    # Shuffle
    random.shuffle(all_data)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")

    # Stats
    from collections import Counter

    domain_counts = Counter(e["domain"] for e in all_data)
    hep_count = sum(1 for e in all_data if e.get("is_hep"))

    print(f"\n[router-data] total: {len(all_data)} training examples")
    print(f"  domains: {dict(domain_counts)}")
    print(f"  HEP-labeled: {hep_count}")
    print(f"  saved to: {output_path}")


if __name__ == "__main__":
    main()
