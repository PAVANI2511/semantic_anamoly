"""
Script to create the expanded 12-document multi-domain benchmark dataset.
Encompasses Finance, Medicine, Climate, ML, Legal, Astronomy, Cybersecurity,
Macroeconomics, Quantum Computing, Immunology, Renewable Energy, and Aerospace.
"""

import json
from pathlib import Path
import pandas as pd

benchmark_data = {
    "benchmark_metadata": {
        "dataset_name": "DGSA Multi-Domain Semantic Anomaly Benchmark v2.0",
        "is_synthetic": True,
        "description": "Gold-standard research benchmark featuring 12 diverse domain documents with curated semantic anomalies and verified character spans.",
        "disclaimer": "Curated for rigorous algorithmic verification and zero-leakage document-level baseline comparison.",
        "version": "2.0",
        "total_documents": 12
    },
    "documents": [
        {
            "doc_id": "DOC_01",
            "title": "Climate Science Homeostasis",
            "domain": "environmental_science",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Global climate change represents one of the most pressing challenges confronting modern civilization.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Rising concentrations of atmospheric greenhouse gases have trapped excess thermal energy within the Earth's planetary systems.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Scientific observations collected across satellite arrays confirm widespread retreats of polar ice sheets and mountain glaciers.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "Average surface temperatures across both continental and oceanic zones have steadily escalated over the preceding century.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 4, "text": "Ocean acidification has accelerated as marine waters absorb substantial quantities of anthropogenic carbon dioxide emissions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "These systemic environmental perturbations trigger shifts in meteorological precipitation patterns and heighten the frequency of severe weather events.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Mitigating these multifaceted ecological impacts necessitates international policy cooperation and rapid transition towards renewable energy matrices.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 7, "text": "Comprehensive decarbonization strategies across energy, transport, and agricultural sectors are imperative to preserve planetary homeostasis.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_02",
            "title": "Quantum Computing Superposition",
            "domain": "quantum_physics",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Quantum computing leverages the fundamental principles of quantum mechanics to perform computational calculations exponentially faster than classical Turing architectures.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Unlike classical bits which remain constrained to binary states of zero or one, quantum bits or qubits exploit superposition to exist simultaneously in combinations of both eigenstates.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Quantum entanglement creates non-local correlations between distinct qubits, establishing computational state spaces of immense dimensional complexity.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "To bake an artisan chocolate cake, preheat your convection oven to 350 degrees and whisk three fresh eggs with refined cane sugar.", "is_anomaly": 1, "category": "topic_deviation", "token_spans": [{"text": "bake", "start_char": 3, "end_char": 7}, {"text": "chocolate", "start_char": 19, "end_char": 28}, {"text": "cake", "start_char": 29, "end_char": 33}, {"text": "oven", "start_char": 58, "end_char": 62}]},
                {"sentence_id": 4, "text": "Decoherence represents the primary engineering obstacle in scalable quantum processor design, where thermal noise disrupts fragile quantum wavefunctions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Advanced quantum error correction codes distribute logical information across manifolds of physical qubits to preserve coherence during complex gate operations.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Future fault-tolerant quantum processors hold transformative potential for drug discovery, material science simulations, and integer factorization.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_03",
            "title": "Apex Technologies Financial Earnings",
            "domain": "finance",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Apex Technologies recorded extraordinary commercial performance throughout the third fiscal quarter of 2026.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Total enterprise revenues increased by 38 percent year-over-year, propelled by enterprise adoption of cloud analytics solutions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Gross operating profit margins expanded across all geographic operating sectors due to rigorous supply chain efficiencies.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "Operating cash flows reached record liquidity levels, empowering the company to fund ambitious research and development initiatives.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 4, "text": "The company experienced total financial failure with overall enterprise revenues that decreased drastically by 40 percent during the exact same period.", "is_anomaly": 1, "category": "contradiction", "token_spans": [{"text": "failure", "start_char": 30, "end_char": 37}, {"text": "decreased", "start_char": 86, "end_char": 95}]},
                {"sentence_id": 5, "text": "Executive leadership announced an enhanced dividend payout schedule reflecting robust confidence in sustained multi-year balance sheet momentum.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Customer retention metrics among Fortune 500 accounts remained at an exceptional 99 percent compliance rate.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_04",
            "title": "Acute Myocardial Infarction Clinical Case",
            "domain": "medicine",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "The patient presented to the acute cardiology ward with persistent retrosternal chest tightness and shortness of breath upon modest exertion.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Twelve-lead electrocardiogram readings demonstrated marked ST-segment elevations localized in the anterior precordial leads V1 through V4.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Emergency serum cardiac troponin biomarkers were substantially elevated, corroborating an evolving myocardial infarction.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "The patient was immediately administered aspirin, intravenous heparin, and sublingual nitroglycerin while awaiting urgent catheterization.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 4, "text": "Detective Miller reviewed the surveillance footage from the bank robbery to identify the armed getaway vehicle.", "is_anomaly": 1, "category": "entity_inconsistency", "token_spans": [{"text": "Detective", "start_char": 0, "end_char": 9}, {"text": "Miller", "start_char": 10, "end_char": 16}, {"text": "robbery", "start_char": 64, "end_char": 71}]},
                {"sentence_id": 5, "text": "Coronary angiography revealed severe critical occlusion of the proximal left anterior descending coronary artery.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Percutaneous coronary intervention was executed smoothly with deployment of a single drug-eluting stent and restoration of normal coronary perfusion.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 7, "text": "Post-procedure recovery remained uneventful with gradual normalization of hemodynamics and cardiac enzyme levels.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_05",
            "title": "Mars Geological Survey Rover",
            "domain": "astronomy",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "The robotic exploration rover completed its autonomous geological survey across the arid terrain of Jezero Crater on Mars.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "High-resolution spectroscopic cameras captured sedimentary layering that indicates the ancient presence of lacustrine waterways.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Subsurface radar instrumentation scanned underlying bedrock structures to map potential mineral formations.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "The rover team celebrated as warm liquid rainwater fell gently upon the lush green Martian forests under the tropical blue sky.", "is_anomaly": 1, "category": "attribute_inconsistency", "token_spans": [{"text": "rainwater", "start_char": 41, "end_char": 50}, {"text": "lush", "start_char": 71, "end_char": 75}, {"text": "forests", "start_char": 82, "end_char": 89}, {"text": "tropical", "start_char": 107, "end_char": 115}]},
                {"sentence_id": 4, "text": "Geological core drill samples were hermetically sealed into titanium sample tubes for retrieval by future rendezvous missions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Thermal sensors recorded surface temperatures plummeting past minus eighty degrees Celsius as the Martian night set in.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Telemetry confirms all subsystems remain nominal while solar arrays recharge internal power reserves.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_06",
            "title": "Cybersecurity APT Threat Vector Analysis",
            "domain": "cybersecurity",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Security analysts detected sophisticated lateral movement within the perimeter network originating from a compromised VPN gateway.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "The adversary leveraged living-off-the-land binaries including PowerShell and WMI to evade endpoint detection systems.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Privilege escalation was achieved by exploiting an unpatched vulnerability in the local kernel authentication subsystem.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "Pruning tomato vines in early summer requires sterile shears to maximize sunlight absorption across healthy garden plots.", "is_anomaly": 1, "category": "contextual_inconsistency", "token_spans": [{"text": "tomato", "start_char": 8, "end_char": 14}, {"text": "vines", "start_char": 15, "end_char": 20}, {"text": "garden", "start_char": 100, "end_char": 106}]},
                {"sentence_id": 4, "text": "Encrypted command-and-control beaconing was observed tunneling outbound traffic through pseudo-legitimate HTTPS headers.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Incident response teams severed the affected subnets and initiated credentials revocation across enterprise active directory domains.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Forensic disk artifacts were preserved for reverse engineering of the zero-day payload.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_07",
            "title": "Software Licensing Intellectual Property",
            "domain": "legal",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "The Licensor hereby grants to the Licensee a perpetual, worldwide, exclusive license to utilize and distribute the proprietary software.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "All patent rights, trade secret protections, and intellectual properties generated under this agreement shall vest solely in the Licensee.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "The Licensor warrants that the delivered source code is devoid of unauthorized third-party copyleft components.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "The Licensor does not grant any rights whatsoever and explicitly retains absolute exclusive ownership over all software assets.", "is_anomaly": 1, "category": "contradiction", "token_spans": [{"text": "not", "start_char": 18, "end_char": 21}, {"text": "grant", "start_char": 22, "end_char": 27}, {"text": "retains", "start_char": 64, "end_char": 71}]},
                {"sentence_id": 4, "text": "Both corporate parties agree that disputes shall be settled through binding arbitration under the rules of the International Chamber of Commerce.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Confidentiality obligations shall endure for a duration of five calendar years following contract termination.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Any unauthorized commercial sublicensing constitutes a material breach triggering immediate statutory damages.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_08",
            "title": "Deep Neural Network Optimization",
            "domain": "machine_learning",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Transformer architectures utilize multi-head self-attention mechanisms to capture long-range token dependencies in natural language sequences.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Layer normalization and residual skip connections stabilize gradient propagation across hundreds of stacked attention layers.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Adaptive learning rate optimizers such as AdamW dynamically scale weight updates based on running second moment estimates.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "A standard major league baseball diamond features ninety feet between consecutive bases with a pitcher mound elevated ten inches.", "is_anomaly": 1, "category": "topic_deviation", "token_spans": [{"text": "baseball", "start_char": 26, "end_char": 34}, {"text": "diamond", "start_char": 35, "end_char": 42}, {"text": "pitcher", "start_char": 95, "end_char": 102}]},
                {"sentence_id": 4, "text": "Dropout regularization and weight decay prevent overparameterized networks from overfitting to idiosyncratic training data distributions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Rotary position embeddings inject relative positional semantics directly into the attention query-key inner products.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Empirical evaluations demonstrate that scaling laws govern validation perplexity improvements proportional to compute expenditure.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_09",
            "title": "Federal Reserve Monetary Policy",
            "domain": "macroeconomics",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "The Federal Open Market Committee convened to evaluate inflationary pressures and national labor market dynamics.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "To curb persistent core inflation, governors voted unanimously to raise the benchmark federal funds rate by fifty basis points to 5.50 percent.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Treasury yield curves responded with modest flattening across intermediate ten-year maturities.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "The Federal Reserve unexpectedly reduced the benchmark federal funds rate down to 0.25 percent during the same afternoon meeting.", "is_anomaly": 1, "category": "contradiction", "token_spans": [{"text": "reduced", "start_char": 33, "end_char": 40}, {"text": "0.25", "start_char": 82, "end_char": 86}]},
                {"sentence_id": 4, "text": "Quantitative tightening policies proceeded in parallel with scheduled reductions in agency mortgage-backed securities holdings.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Central bank communication emphasized data-dependent decision making for subsequent monetary policy adjustments.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Commercial lending institutions tightened underwriting criteria in response to elevated capital reserve requirements.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_10",
            "title": "Monoclonal Antibody Therapeutics",
            "domain": "immunology",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Monoclonal antibody bio-therapeutics bind with extreme stereochemical specificity to extracellular surface receptors.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Upon binding targeted oncogenic epitopes, they impede downstream intracellular kinase signaling cascades.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Fc domain engagement recruits natural killer cells to execute antibody-dependent cellular cytotoxicity.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "Complement-dependent cytotoxicity further enhances lysis of targeted malignant tissue through membrane attack complex formation.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 4, "text": "Pharmacokinetic studies demonstrate that humanized IgG1 scaffolds display prolonged systemic half-lives exceeding twenty days.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Affinity maturation workflows optimize complementarity-determining regions to achieve sub-nanomolar dissociation constants.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Clinical trial cohorts demonstrated significant progression-free survival improvements with manageable adverse immunogenicity profiles.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_11",
            "title": "Renewable Power Grid Synchronization",
            "domain": "energy_systems",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "High penetration of renewable generation challenges traditional transmission grid stability and rotational inertia reserves.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Grid-forming inverters synthesize virtual synchronous voltage waveforms to stabilize system AC frequency deviations.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "Utility-scale lithium iron phosphate battery storage systems absorb intermittent surplus generation during peak solar daylight hours.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "The photovoltaic solar panels generated peak midday solar electricity at midnight under absolute total darkness with zero light.", "is_anomaly": 1, "category": "attribute_inconsistency", "token_spans": [{"text": "midnight", "start_char": 74, "end_char": 82}, {"text": "darkness", "start_char": 103, "end_char": 111}]},
                {"sentence_id": 4, "text": "Dynamic reactive power compensation stabilizes distribution voltage profiles across long transmission corridors.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Automated demand response platforms shed non-critical industrial electrical loads during abrupt renewable production drops.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "High-voltage direct current interties provide robust bulk power transmission capacity across isolated regional synchronous zones.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        },
        {
            "doc_id": "DOC_12",
            "title": "Cryogenic Rocket Propulsion Dynamics",
            "domain": "aerospace",
            "is_synthetic": True,
            "sentences": [
                {"sentence_id": 0, "text": "Staged-combustion rocket engines burn liquid oxygen and liquid hydrogen at chamber pressures exceeding twenty megapascals.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 1, "text": "Pre-burners drive high-speed turbopumps delivering cryogenic propellants into regenerative cooling channels lining the combustion chamber.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 2, "text": "The convergent-divergent de Laval bell nozzle accelerates expanding exhaust gases to hypersonic exit velocities.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 3, "text": "Captain Edward hoisted the mainsail on the luxury wooden yacht to catch the Mediterranean breeze near the harbor.", "is_anomaly": 1, "category": "entity_inconsistency", "token_spans": [{"text": "Edward", "start_char": 8, "end_char": 14}, {"text": "mainsail", "start_char": 27, "end_char": 35}, {"text": "yacht", "start_char": 57, "end_char": 62}]},
                {"sentence_id": 4, "text": "Combustion instability is suppressed via acoustic resonators and optimized coaxial swirl injector placement.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 5, "text": "Specific impulse measurements verify nominal propulsion performance matching theoretical thermodynamic expansion predictions.", "is_anomaly": 0, "category": "normal", "token_spans": []},
                {"sentence_id": 6, "text": "Gimbaled hydraulic actuators provide active thrust vector control to steer the launch vehicle through atmospheric ascent.", "is_anomaly": 0, "category": "normal", "token_spans": []}
            ]
        }
    ]
}

# Save JSON
out_json = Path("data/datasets/research_benchmark.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)
print(f"Saved {out_json} with {len(benchmark_data['documents'])} documents.")

# Also update synthetic_benchmark.json with the same comprehensive data so both work seamlessly
with open(Path("data/datasets/synthetic_benchmark.json"), "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)

# Build CSV
rows = []
for doc in benchmark_data["documents"]:
    doc_id = doc["doc_id"]
    domain = doc["domain"]
    title = doc["title"]
    for s in doc["sentences"]:
        rows.append({
            "doc_id": doc_id,
            "title": title,
            "domain": domain,
            "sentence_id": s["sentence_id"],
            "text": s["text"],
            "is_anomaly": s["is_anomaly"],
            "category": s["category"],
            "token_spans": json.dumps(s["token_spans"])
        })

df = pd.DataFrame(rows)
df.to_csv("data/datasets/research_benchmark.csv", index=False)
df.to_csv("data/datasets/synthetic_benchmark.csv", index=False)
print(f"Saved CSV with {len(df)} total sentences ({df['is_anomaly'].sum()} anomalies).")

