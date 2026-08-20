"""Concept seed catalog: broad, structured topic list across all required domains.

Generates seeds/concept_catalog.json — a list of {domain, subdomain, concept,
difficulty_hint, task_types} entries that drive generation. Designed for diversity and
coverage rather than volume. Each concept will later be expanded into multiple
representations and difficulty levels by the generation pipeline.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "distilled_corpus" / "seeds" / "concept_catalog.json"

# (domain, subdomain, [concepts...])
TREE = {
    "mathematics": {
        "arithmetic": ["order of operations", "fractions", "decimals", "percentages", "ratios and proportions"],
        "algebra": ["linear equations", "quadratic equations", "polynomials", "inequalities", "exponents and logarithms", "sequences and series"],
        "geometry": ["Euclidean geometry", "triangles", "circles", "polygons", "coordinate geometry", "transformations"],
        "trigonometry": ["trig functions", "trig identities", "law of sines and cosines", "unit circle"],
        "calculus": ["limits", "continuity", "derivatives", "integration", "Taylor series", "multivariable calculus", "differential equations basics"],
        "linear_algebra": ["vectors", "matrices", "linear transformations", "eigenvalues and eigenvectors", "vector spaces", "SVD"],
        "probability": ["probability axioms", "Bayes theorem", "random variables", "distributions", "expectation and variance", "law of large numbers", "central limit theorem"],
        "statistics": ["descriptive statistics", "hypothesis testing", "confidence intervals", "regression", "correlation", "Bayesian inference"],
        "discrete_math": ["logic", "set theory", "combinatorics", "graph theory", "recurrence relations"],
        "number_theory": ["primes", "modular arithmetic", "gcd and lcm", "Fermat's little theorem"],
    },
    "physics": {
        "mechanics": ["Newton's laws", "kinematics", "energy and work", "momentum", "rotational motion", "oscillations", "fluid mechanics basics"],
        "thermodynamics": ["laws of thermodynamics", "entropy", "heat engines", "Boltzmann distribution"],
        "electromagnetism": ["Coulomb's law", "electric fields", "magnetic fields", "Maxwell's equations", "induction"],
        "optics": ["reflection and refraction", "lenses and mirrors", "interference and diffraction", "polarization"],
        "quantum_mechanics": ["wave-particle duality", "Schrödinger equation", "uncertainty principle", "quantum numbers"],
        "relativity": ["special relativity", "Lorentz transformations", "general relativity basics", "spacetime"],
        "statistical_mechanics": ["ensembles", "partition function", "Maxwell-Boltzmann statistics"],
        "astrophysics": ["stellar evolution", "black holes", "cosmological expansion", "Hertzsprung-Russell diagram"],
    },
    "chemistry": {
        "general_chemistry": ["atomic structure", "periodic table", "chemical bonding", "stoichiometry", "states of matter", "acids and bases"],
        "organic_chemistry": ["functional groups", "isomerism", "reaction mechanisms", "stereochemistry", "aromaticity"],
        "inorganic_chemistry": ["coordination compounds", "transition metals", "crystal field theory"],
        "physical_chemistry": ["thermodynamics of reactions", "kinetics", "quantum chemistry basics", "electrochemistry"],
        "biochemistry": ["proteins", "enzymes", "metabolism", "DNA and RNA", "carbohydrates and lipids"],
        "materials_chemistry": ["polymers", "semiconductors", "nanomaterials", "crystal structures"],
    },
    "biology": {
        "cell_biology": ["cell structure", "membranes", "cell cycle", "mitosis and meiosis", "cellular respiration", "photosynthesis"],
        "molecular_biology": ["central dogma", "transcription", "translation", "PCR", "gene regulation"],
        "genetics": ["Mendelian genetics", "linkage and recombination", "mutations", "population genetics"],
        "evolution": ["natural selection", "genetic drift", "speciation", "molecular clock"],
        "ecology": ["food webs", "nutrient cycles", "population dynamics", "biodiversity"],
        "microbiology": ["bacteria", "viruses", "antibiotics", "fermentation"],
        "physiology": ["nervous system", "circulatory system", "immune system", "endocrine system"],
        "neuroscience": ["neurons", "action potentials", "synapses", "neural circuits"],
        "biotechnology": ["CRISPR", "recombinant DNA", "bioreactors", "synthetic biology"],
    },
    "earth_space": {
        "geology": ["plate tectonics", "minerals", "rock cycle", "earthquakes", "volcanism"],
        "climatology": ["climate systems", "greenhouse effect", "El Nino", "ice ages"],
        "oceanography": ["ocean currents", "tides", "marine ecosystems"],
        "planetary_science": ["solar system formation", "planetary atmospheres", "moons"],
        "astronomy": ["stars", "galaxies", "telescopes", "exoplanets", "stellar classification"],
        "cosmology": ["Big Bang", "cosmic microwave background", "dark matter", "dark energy", "expansion of the universe"],
    },
    "engineering": {
        "electrical": ["Ohm's law", "Kirchhoff's laws", "capacitors and inductors", "AC circuits", "semiconductor devices"],
        "mechanical": ["statics", "dynamics", "materials strength", "thermodynamic cycles", "heat transfer"],
        "civil": ["structural analysis", "concrete and steel", "hydraulics", "surveying"],
        "chemical": ["mass balance", "reactor design", "separation processes", "process control"],
        "control_systems": ["feedback", "PID control", "stability", "transfer functions"],
        "robotics": ["kinematics", "dynamics", "sensors and actuators", "motion planning"],
        "materials_science": ["crystal defects", "phase diagrams", "mechanical properties", "failure analysis"],
    },
    "technology": {
        "programming": ["variables and types", "control flow", "functions", "recursion", "error handling", "OOP", "functional programming"],
        "algorithms": ["sorting", "searching", "dynamic programming", "greedy algorithms", "divide and conquer", "graph algorithms", "complexity analysis"],
        "data_structures": ["arrays", "linked lists", "stacks and queues", "trees", "hash tables", "heaps", "graphs"],
        "databases": ["relational model", "SQL", "normalization", "indexes", "transactions", "NoSQL", "ACID"],
        "operating_systems": ["processes and threads", "scheduling", "memory management", "file systems", "concurrency"],
        "networking": ["OSI model", "TCP/IP", "routing", "DNS", "HTTP", "sockets"],
        "distributed_systems": ["consensus", "CAP theorem", "replication", "clocks and ordering", "mapreduce"],
        "cloud_computing": ["virtualization", "containers", "serverless", "scalability", "object storage"],
        "cybersecurity": ["authentication", "encryption", "hashing", "attacks and defenses", "PKI", "threat modeling"],
        "compilers": ["lexing", "parsing", "semantic analysis", "optimization", "code generation"],
        "computer_architecture": ["instruction sets", "pipelining", "cache hierarchy", "parallelism", "GPU architecture"],
        "software_engineering": ["design patterns", "testing", "version control", "code review", "refactoring"],
        "system_design": ["load balancing", "caching", "message queues", "sharding", "rate limiting"],
        "machine_learning": ["supervised learning", "linear regression", "logistic regression", "decision trees", "ensembles", "clustering", "overfitting and regularization", "cross-validation"],
        "deep_learning": ["neural networks", "backpropagation", "CNNs", "RNNs", "attention", "transformers", "normalization", "optimizers", "regularization (dropout)"],
        "reinforcement_learning": ["MDPs", "Q-learning", "policy gradient", "PPO basics"],
        "computer_vision": ["image filtering", "edge detection", "object detection", "image segmentation"],
        "nlp": ["tokenization", "embeddings", "language models", "seq2seq", "transformer architectures"],
        "llms": ["pretraining", "fine-tuning", "RLHF", "context length", "attention mechanisms", "KV cache", "scaling laws", "distillation"],
        "ai_agents": ["planning", "tool use", "memory", "ReAct", "multi-agent systems"],
    },
    "general_knowledge": {
        "history": ["ancient civilizations", "Roman empire", "middle ages", "renaissance", "industrial revolution", "world wars", "cold war", "decolonization"],
        "geography": ["continents", "major rivers", "mountain ranges", "deserts", "oceans", "countries and capitals"],
        "politics": ["democracy", "separation of powers", "electoral systems", "constitutions", "international organizations"],
        "economics": ["supply and demand", "GDP", "inflation", "monetary policy", "trade", "market structures"],
        "sociology": ["social institutions", "culture", "socialization", "inequality", "demographics"],
        "philosophy": ["epistemology", "ethics", "logic", "metaphysics", "major philosophers"],
        "literature": ["genres", "literary devices", "major authors", "narrative structure"],
        "law": ["contract law", "criminal law", "constitutional law", "intellectual property", "international law"],
        "education": ["learning theories", "pedagogy", "assessment", "curriculum design"],
        "everyday": ["nutrition basics", "personal finance", "health and hygiene", "transportation", "communication"],
    },
}

DIFFICULTIES = [1, 2, 3, 4, 5]
# task type templates per concept (multi-representation per the spec section 10)
TASK_TYPES = [
    "definition",
    "beginner_explanation",
    "intermediate_explanation",
    "university_explanation",
    "expert_explanation",
    "mechanism_or_why",
    "worked_example",
    "comparison",
    "common_misconceptions",
    "interdisciplinary_connection",
    "encyclopedia_article",
    "qa_problem_solving",
]


def build():
    seeds = []
    sid = 0
    for domain, subs in TREE.items():
        for sub, concepts in subs.items():
            for c in concepts:
                sid += 1
                seeds.append({
                    "id": f"seed-{sid:05d}",
                    "domain": domain,
                    "subdomain": sub,
                    "concept": c,
                    "difficulty_levels": DIFFICULTIES,
                    "task_types": TASK_TYPES,
                    "knowledge_type": "time_invariant",
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(seeds, indent=2))
    print(f"{len(seeds)} concept seeds across {len(TREE)} domains -> {OUT}")
    # also print domain counts
    from collections import Counter
    cnt = Counter(s["domain"] for s in seeds)
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
