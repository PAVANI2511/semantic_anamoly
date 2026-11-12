"""
Script to create the expanded 20-document multi-domain research benchmark dataset.
Encompasses:
- Environmental Science / Climate
- Quantum Physics
- Corporate Finance
- Cardiology / Medicine
- Planetary Astronomy
- Cybersecurity / APT
- Legal Intellectual Property
- Deep Learning / ML Optimization
- Macroeconomics
- Oncology / Immunology (Control / Normal)
- Renewable Energy Systems
- Aerospace / Rocket Propulsion
- Supply Chain Logistics (Temporal Inconsistency)
- Cardiovascular Pharmacology (Numerical Inconsistency)
- Autonomous Vehicle Perception (Topic Deviation)
- Constitutional Law (Control / Normal)
- Semiconductor Lithography (Contextual Inconsistency)
- Diplomatic History (Entity Inconsistency)
- Commercial Avionics (Attribute Inconsistency)
- Hydropower Engineering (Temporal Inconsistency)

All 7 anomaly categories represented + 2 Pure Normal/Control documents.
"""

import json
from pathlib import Path
import pandas as pd


def create_span(sentence_text: str, target_phrase: str):
    """Finds exact character offsets for a phrase within a sentence."""
    start = sentence_text.find(target_phrase)
    if start == -1:
        raise ValueError(f"Target '{target_phrase}' not found in '{sentence_text}'")
    return {
        "text": target_phrase,
        "start_char": start,
        "end_char": start + len(target_phrase)
    }


# Raw document definitions
raw_docs = [
    {
        "doc_id": "DOC_01",
        "title": "Climate Science Homeostasis",
        "domain": "environmental_science",
        "sentences": [
            ("Global climate change represents one of the most pressing challenges confronting modern civilization.", 0, "normal", []),
            ("Rising concentrations of atmospheric greenhouse gases have trapped excess thermal energy within the Earth's planetary systems.", 0, "normal", []),
            ("Scientific observations collected across satellite arrays confirm widespread retreats of polar ice sheets and mountain glaciers.", 0, "normal", []),
            ("Average surface temperatures across both continental and oceanic zones have steadily escalated over the preceding century.", 0, "normal", []),
            ("Ocean acidification has accelerated as marine waters absorb substantial quantities of anthropogenic carbon dioxide emissions.", 0, "normal", []),
            ("These systemic environmental perturbations trigger shifts in meteorological precipitation patterns and heighten the frequency of severe weather events.", 0, "normal", []),
            ("Mitigating these multifaceted ecological impacts necessitates international policy cooperation and rapid transition towards renewable energy matrices.", 0, "normal", []),
            ("Comprehensive decarbonization strategies across energy, transport, and agricultural sectors are imperative to preserve planetary homeostasis.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_02",
        "title": "Quantum Computing Superposition",
        "domain": "quantum_physics",
        "sentences": [
            ("Quantum computing leverages the fundamental principles of quantum mechanics to perform computational calculations exponentially faster than classical Turing architectures.", 0, "normal", []),
            ("Unlike classical bits which remain constrained to binary states of zero or one, quantum bits or qubits exploit superposition to exist simultaneously in combinations of both eigenstates.", 0, "normal", []),
            ("Quantum entanglement creates non-local correlations between distinct qubits, establishing computational state spaces of immense dimensional complexity.", 0, "normal", []),
            ("To bake an artisan chocolate cake, preheat your convection oven to 350 degrees and whisk three fresh eggs with refined cane sugar.", 1, "topic_deviation", ["bake", "chocolate cake", "oven"]),
            ("Decoherence represents the primary engineering obstacle in scalable quantum processor design, where thermal noise disrupts fragile quantum wavefunctions.", 0, "normal", []),
            ("Advanced quantum error correction codes distribute logical information across manifolds of physical qubits to preserve coherence during complex gate operations.", 0, "normal", []),
            ("Future fault-tolerant quantum processors hold transformative potential for drug discovery, material science simulations, and integer factorization.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_03",
        "title": "Apex Technologies Financial Earnings",
        "domain": "finance",
        "sentences": [
            ("Apex Technologies recorded extraordinary commercial performance throughout the third fiscal quarter of 2026.", 0, "normal", []),
            ("Total enterprise revenues increased by 38 percent year-over-year, propelled by enterprise adoption of cloud analytics solutions.", 0, "normal", []),
            ("Gross operating profit margins expanded across all geographic operating sectors due to rigorous supply chain efficiencies.", 0, "normal", []),
            ("Operating cash flows reached record liquidity levels, empowering the company to fund ambitious research and development initiatives.", 0, "normal", []),
            ("The company experienced total financial failure with overall enterprise revenues that decreased drastically by 40 percent during the exact same period.", 1, "contradiction", ["financial failure", "decreased drastically"]),
            ("Executive leadership announced an enhanced dividend payout schedule reflecting robust confidence in sustained multi-year balance sheet momentum.", 0, "normal", []),
            ("Customer retention metrics among Fortune 500 accounts remained at an exceptional 99 percent compliance rate.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_04",
        "title": "Acute Myocardial Infarction Clinical Case",
        "domain": "medicine",
        "sentences": [
            ("The patient presented to the acute cardiology ward with persistent retrosternal chest tightness and shortness of breath upon modest exertion.", 0, "normal", []),
            ("Twelve-lead electrocardiogram readings demonstrated marked ST-segment elevations localized in the anterior precordial leads V1 through V4.", 0, "normal", []),
            ("Emergency serum cardiac troponin biomarkers were substantially elevated, corroborating an evolving myocardial infarction.", 0, "normal", []),
            ("The patient was immediately administered aspirin, intravenous heparin, and sublingual nitroglycerin while awaiting urgent catheterization.", 0, "normal", []),
            ("Detective Miller reviewed the surveillance footage from the bank robbery to identify the armed getaway vehicle.", 1, "entity_inconsistency", ["Detective Miller", "bank robbery"]),
            ("Coronary angiography revealed severe critical occlusion of the proximal left anterior descending coronary artery.", 0, "normal", []),
            ("Percutaneous coronary intervention was executed smoothly with deployment of a single drug-eluting stent and restoration of normal coronary perfusion.", 0, "normal", []),
            ("Post-procedure recovery remained uneventful with gradual normalization of hemodynamics and cardiac enzyme levels.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_05",
        "title": "Mars Geological Survey Rover",
        "domain": "astronomy",
        "sentences": [
            ("The robotic exploration rover completed its autonomous geological survey across the arid terrain of Jezero Crater on Mars.", 0, "normal", []),
            ("High-resolution spectroscopic cameras captured sedimentary layering that indicates the ancient presence of lacustrine waterways.", 0, "normal", []),
            ("Subsurface radar instrumentation scanned underlying bedrock structures to map potential mineral formations.", 0, "normal", []),
            ("The rover team celebrated as warm liquid rainwater fell gently upon the lush green Martian forests under the tropical blue sky.", 1, "attribute_inconsistency", ["rainwater", "lush green Martian forests", "tropical"]),
            ("Geological core drill samples were hermetically sealed into titanium sample tubes for retrieval by future rendezvous missions.", 0, "normal", []),
            ("Thermal sensors recorded surface temperatures plummeting past minus eighty degrees Celsius as the Martian night set in.", 0, "normal", []),
            ("Telemetry confirms all subsystems remain nominal while solar arrays recharge internal power reserves.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_06",
        "title": "Cybersecurity APT Threat Vector Analysis",
        "domain": "cybersecurity",
        "sentences": [
            ("Security analysts detected sophisticated lateral movement within the perimeter network originating from a compromised VPN gateway.", 0, "normal", []),
            ("The adversary leveraged living-off-the-land binaries including PowerShell and WMI to evade endpoint detection systems.", 0, "normal", []),
            ("Privilege escalation was achieved by exploiting an unpatched vulnerability in the local kernel authentication subsystem.", 0, "normal", []),
            ("Pruning tomato vines in early summer requires sterile shears to maximize sunlight absorption across healthy garden plots.", 1, "contextual_inconsistency", ["tomato vines", "garden plots"]),
            ("Encrypted command-and-control beaconing was observed tunneling outbound traffic through pseudo-legitimate HTTPS headers.", 0, "normal", []),
            ("Incident response teams severed the affected subnets and initiated credentials revocation across enterprise active directory domains.", 0, "normal", []),
            ("Forensic disk artifacts were preserved for reverse engineering of the zero-day payload.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_07",
        "title": "Software Licensing Intellectual Property",
        "domain": "legal",
        "sentences": [
            ("The Licensor hereby grants to the Licensee a perpetual, worldwide, exclusive license to utilize and distribute the proprietary software.", 0, "normal", []),
            ("All patent rights, trade secret protections, and intellectual properties generated under this agreement shall vest solely in the Licensee.", 0, "normal", []),
            ("The Licensor warrants that the delivered source code is devoid of unauthorized third-party copyleft components.", 0, "normal", []),
            ("The Licensor does not grant any rights whatsoever and explicitly retains absolute exclusive ownership over all software assets.", 1, "contradiction", ["does not grant", "retains absolute exclusive ownership"]),
            ("Both corporate parties agree that disputes shall be settled through binding arbitration under the rules of the International Chamber of Commerce.", 0, "normal", []),
            ("Confidentiality obligations shall endure for a duration of five calendar years following contract termination.", 0, "normal", []),
            ("Any unauthorized commercial sublicensing constitutes a material breach triggering immediate statutory damages.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_08",
        "title": "Deep Neural Network Optimization",
        "domain": "machine_learning",
        "sentences": [
            ("Transformer architectures utilize multi-head self-attention mechanisms to capture long-range token dependencies in natural language sequences.", 0, "normal", []),
            ("Layer normalization and residual skip connections stabilize gradient propagation across hundreds of stacked attention layers.", 0, "normal", []),
            ("Adaptive learning rate optimizers such as AdamW dynamically scale weight updates based on running second moment estimates.", 0, "normal", []),
            ("A standard major league baseball diamond features ninety feet between consecutive bases with a pitcher mound elevated ten inches.", 1, "topic_deviation", ["baseball diamond", "pitcher mound"]),
            ("Dropout regularization and weight decay prevent overparameterized networks from overfitting to idiosyncratic training data distributions.", 0, "normal", []),
            ("Rotary position embeddings inject relative positional semantics directly into the attention query-key inner products.", 0, "normal", []),
            ("Empirical evaluations demonstrate that scaling laws govern validation perplexity improvements proportional to compute expenditure.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_09",
        "title": "Federal Reserve Monetary Policy",
        "domain": "macroeconomics",
        "sentences": [
            ("The Federal Open Market Committee convened to evaluate inflationary pressures and national labor market dynamics.", 0, "normal", []),
            ("To curb persistent core inflation, governors voted unanimously to raise the benchmark federal funds rate by fifty basis points to 5.50 percent.", 0, "normal", []),
            ("Treasury yield curves responded with modest flattening across intermediate ten-year maturities.", 0, "normal", []),
            ("The Federal Reserve unexpectedly reduced the benchmark federal funds rate down to 0.25 percent during the same afternoon meeting.", 1, "contradiction", ["reduced the benchmark", "0.25 percent"]),
            ("Quantitative tightening policies proceeded in parallel with scheduled reductions in agency mortgage-backed securities holdings.", 0, "normal", []),
            ("Central bank communication emphasized data-dependent decision making for subsequent monetary policy adjustments.", 0, "normal", []),
            ("Commercial lending institutions tightened underwriting criteria in response to elevated capital reserve requirements.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_10",
        "title": "Monoclonal Antibody Therapeutics",
        "domain": "immunology",
        "sentences": [
            ("Monoclonal antibody bio-therapeutics bind with extreme stereochemical specificity to extracellular surface receptors.", 0, "normal", []),
            ("Upon binding targeted oncogenic epitopes, they impede downstream intracellular kinase signaling cascades.", 0, "normal", []),
            ("Fc domain engagement recruits natural killer cells to execute antibody-dependent cellular cytotoxicity.", 0, "normal", []),
            ("Complement-dependent cytotoxicity further enhances lysis of targeted malignant tissue through membrane attack complex formation.", 0, "normal", []),
            ("Pharmacokinetic studies demonstrate that humanized IgG1 scaffolds display prolonged systemic half-lives exceeding twenty days.", 0, "normal", []),
            ("Affinity maturation workflows optimize complementarity-determining regions to achieve sub-nanomolar dissociation constants.", 0, "normal", []),
            ("Clinical trial cohorts demonstrated significant progression-free survival improvements with manageable adverse immunogenicity profiles.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_11",
        "title": "Renewable Power Grid Synchronization",
        "domain": "energy_systems",
        "sentences": [
            ("High penetration of renewable generation challenges traditional transmission grid stability and rotational inertia reserves.", 0, "normal", []),
            ("Grid-forming inverters synthesize virtual synchronous voltage waveforms to stabilize system AC frequency deviations.", 0, "normal", []),
            ("Utility-scale lithium iron phosphate battery storage systems absorb intermittent surplus generation during peak solar daylight hours.", 0, "normal", []),
            ("The photovoltaic solar panels generated peak midday solar electricity at midnight under absolute total darkness with zero light.", 1, "attribute_inconsistency", ["midnight", "darkness", "zero light"]),
            ("Dynamic reactive power compensation stabilizes distribution voltage profiles across long transmission corridors.", 0, "normal", []),
            ("Automated demand response platforms shed non-critical industrial electrical loads during abrupt renewable production drops.", 0, "normal", []),
            ("High-voltage direct current interties provide robust bulk power transmission capacity across isolated regional synchronous zones.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_12",
        "title": "Cryogenic Rocket Propulsion Dynamics",
        "domain": "aerospace",
        "sentences": [
            ("Staged-combustion rocket engines burn liquid oxygen and liquid hydrogen at chamber pressures exceeding twenty megapascals.", 0, "normal", []),
            ("Pre-burners drive high-speed turbopumps delivering cryogenic propellants into regenerative cooling channels lining the combustion chamber.", 0, "normal", []),
            ("The convergent-divergent de Laval bell nozzle accelerates expanding exhaust gases to hypersonic exit velocities.", 0, "normal", []),
            ("Captain Edward hoisted the mainsail on the luxury wooden yacht to catch the Mediterranean breeze near the harbor.", 1, "entity_inconsistency", ["Captain Edward", "mainsail", "luxury wooden yacht"]),
            ("Combustion instability is suppressed via acoustic resonators and optimized coaxial swirl injector placement.", 0, "normal", []),
            ("Specific impulse measurements verify nominal propulsion performance matching theoretical thermodynamic expansion predictions.", 0, "normal", []),
            ("Gimbaled hydraulic actuators provide active thrust vector control to steer the launch vehicle through atmospheric ascent.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_13",
        "title": "Global Semiconductor Supply Chain Lead Times",
        "domain": "supply_chain",
        "sentences": [
            ("Advanced fabrication of three-nanometer silicon wafers requires high-precision extreme ultraviolet photolithography.", 0, "normal", []),
            ("Fabrication lead times regularly span between 16 and 26 weeks from raw boule slicing to final die packaging.", 0, "normal", []),
            ("Finished integrated circuits undergo automated optical inspection before international freight dispatch.", 0, "normal", []),
            ("The cargo freight ship delivered the finished semiconductors on Monday morning, five weeks before the silicon wafers were manufactured in the cleanroom on Friday.", 1, "temporal_inconsistency", ["five weeks before", "manufactured"]),
            ("Robust supply chain logistics buffer against oceanic cargo bottlenecks and transshipment port congestion.", 0, "normal", []),
            ("Just-in-time inventory policies maintain holding costs within designated enterprise operational targets.", 0, "normal", []),
            ("Assembly, testing, and packaging facilities receive processed silicon dies according to strict quarterly production schedules.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_14",
        "title": "Cardiovascular Pharmacology Dosage Regimen",
        "domain": "pharmacology",
        "sentences": [
            ("Intravenous carvedilol administration reduces systemic blood pressure by attenuating alpha-1 and beta-adrenergic receptor stimulation.", 0, "normal", []),
            ("The standard starting clinical dosage for congestive heart failure initiates at 3.125 milligrams twice daily.", 0, "normal", []),
            ("Gradual titration occurs at bi-weekly intervals toward a ceiling maximum target of 25.0 milligrams twice daily.", 0, "normal", []),
            ("The resident physician administered an acute single bolus dose of 3125.0 grams of carvedilol to the neonatal infant patient.", 1, "numerical_inconsistency", ["3125.0 grams", "neonatal infant"]),
            ("Continuous electrocardiographic telemetry ensures absence of bradycardia or high-grade atrioventricular block.", 0, "normal", []),
            ("Renal and hepatic clearance profiles dictate therapeutic drug monitoring frequencies and metabolic clearance rates.", 0, "normal", []),
            ("Patient systolic pressures stabilized within the target therapeutic window of 110 to 125 millimeters of mercury.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_15",
        "title": "Autonomous Vehicle Perception and Tracking",
        "domain": "autonomous_vehicles",
        "sentences": [
            ("Autonomous perception systems fuse point cloud data from LiDAR sensors with multi-camera computer vision feeds.", 0, "normal", []),
            ("Deep neural network backbones classify dynamic road obstacles including pedestrians, cyclists, and moving vehicles.", 0, "normal", []),
            ("Extended Kalman filter trackers predict the trajectory and velocity vectors of surrounding traffic participants.", 0, "normal", []),
            ("Grandmother knits warm wool scarves and colorful winter mittens beside the crackling brick fireplace.", 1, "topic_deviation", ["knits", "wool scarves", "winter mittens", "fireplace"]),
            ("The motion planning module generates collision-free spline trajectories within road boundary constraints.", 0, "normal", []),
            ("Model predictive control algorithms calculate steering angles and longitudinal acceleration commands.", 0, "normal", []),
            ("Redundant drive-by-wire actuators execute vehicle maneuvering with sub-millisecond response latency.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_16",
        "title": "Constitutional Law and Judicial Review",
        "domain": "constitutional_law",
        "sentences": [
            ("The Supreme Court exercises judicial review to determine the constitutionality of legislative statutes and executive decrees.", 0, "normal", []),
            ("Under the seminal doctrine established in Marbury v. Madison, the judiciary possesses paramount authority to interpret legal mandates.", 0, "normal", []),
            ("Strict scrutiny represents the highest judicial standard applied to statutory classifications implicating fundamental constitutional rights.", 0, "normal", []),
            ("To withstand strict scrutiny, the contested government measure must serve a compelling state interest and be narrowly tailored.", 0, "normal", []),
            ("Lower federal appellate courts apply binding judicial precedent established by Supreme Court jurisprudence.", 0, "normal", []),
            ("The principle of stare decisis fosters institutional legitimacy and legal predictability across common law jurisdictions.", 0, "normal", []),
            ("Dissenting opinions elucidate alternative legal interpretations that may inform future constitutional jurisprudence.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_17",
        "title": "Semiconductor EUV Photolithography",
        "domain": "semiconductor_physics",
        "sentences": [
            ("Extreme ultraviolet lithography systems utilize light with a wavelength of 13.5 nanometers to print nanoscale transistor features.", 0, "normal", []),
            ("High-power carbon dioxide lasers pulse into molten tin droplets fifty thousand times per second to generate extreme ultraviolet plasma.", 0, "normal", []),
            ("Highly reflective multilayer molybdenum-silicon mirrors focus the extreme ultraviolet light onto the photoresist-coated wafer.", 0, "normal", []),
            ("Competitive marathon runners hydrate with isotonic electrolyte beverages to maintain cardiovascular endurance during hilly highway courses.", 1, "contextual_inconsistency", ["marathon runners", "electrolyte beverages", "highway courses"]),
            ("Operating under ultra-high vacuum conditions prevents atmospheric gases from absorbing the fragile ultraviolet radiation.", 0, "normal", []),
            ("Precision interferometer sensors adjust optical alignment with picometer accuracy during exposure steps.", 0, "normal", []),
            ("Nanometer-scale resolution enables continued geometric scaling of dense semiconductor logic and memory circuits.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_18",
        "title": "Twentieth-Century Diplomatic History",
        "domain": "diplomatic_history",
        "sentences": [
            ("Following the conclusion of the Second World War, European diplomatic relations underwent profound structural realignment.", 0, "normal", []),
            ("The Marshall Plan channeled substantial economic reconstruction funds across war-torn Western European democracies.", 0, "normal", []),
            ("The North Atlantic Treaty Organization was chartered in 1949 to provide collective security against potential Soviet expansion.", 0, "normal", []),
            ("Galactic Emperor Palpatine dispatched Imperial Star Destroyers to blockade the rebel base on the desert planet Tatooine.", 1, "entity_inconsistency", ["Emperor Palpatine", "Imperial Star Destroyers", "Tatooine"]),
            ("The Warsaw Pact was subsequently established in 1955 as a counterbalancing military alliance among Eastern Bloc nations.", 0, "normal", []),
            ("Diplomatic tensions between opposing ideological blocs crystallized into the enduring geopolitical confrontation of the Cold War.", 0, "normal", []),
            ("Bilateral nuclear arms limitation negotiations throughout the 1970s eventually led to strategic detente agreements.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_19",
        "title": "Commercial Avionics Flight Control Systems",
        "domain": "aviation_engineering",
        "sentences": [
            ("Modern commercial jetliners incorporate digital fly-by-wire flight control computers to modulate aerodynamic control surfaces.", 0, "normal", []),
            ("Triplicate redundant flight management systems process sensor data from pitot tubes, static ports, and inertial reference units.", 0, "normal", []),
            ("Flight envelope protection software limits pilot inputs to prevent aerodynamic stalls, overspeed conditions, or structural over-G stresses.", 0, "normal", []),
            ("The twin-engine turbofan aircraft flew seamlessly through the outer stratosphere propelled entirely by wooden oars rowing through atmospheric air.", 1, "attribute_inconsistency", ["wooden oars", "rowing"]),
            ("Primary flight displays synthesize attitude, airspeed, altitude, and heading information into intuitive graphical interfaces.", 0, "normal", []),
            ("Hydraulic actuators translate electronic servo signals into mechanical deflections of ailerons, elevators, and rudder surfaces.", 0, "normal", []),
            ("Automated autothrottle subsystems maintain commanded cruise Mach numbers during oceanic navigation waypoints.", 0, "normal", [])
        ]
    },
    {
        "doc_id": "DOC_20",
        "title": "Hydroelectric Dam Power Generation",
        "domain": "hydropower_engineering",
        "sentences": [
            ("Hydroelectric dams capture the gravitational potential energy of impounded river water to generate dispatchable electrical energy.", 0, "normal", []),
            ("High-pressure water rushes through penstocks to turn hydraulic Francis turbine runners connected to electric generator rotors.", 0, "normal", []),
            ("Heavy rotating generator coils convert mechanical shaft torque into alternating electric current through electromagnetic induction.", 0, "normal", []),
            ("The hydroelectric dam generated ten gigawatt-hours of clean electricity on Tuesday, three centuries before civil engineers constructed the concrete foundation in the year 2012.", 1, "temporal_inconsistency", ["three centuries before", "2012"]),
            ("Spillway gates regulate reservoir storage levels during seasonal torrential rainfall and snowmelt runoff events.", 0, "normal", []),
            ("Step-up transformers elevate voltage levels to hundreds of kilovolts for low-loss transmission across bulk electrical grids.", 0, "normal", []),
            ("Pumped-storage facilities pump water back to the upper reservoir during off-peak hours to provide grid storage capacity.", 0, "normal", [])
        ]
    }
]


def build_and_save_dataset():
    documents = []
    for doc_data in raw_docs:
        doc_id = doc_data["doc_id"]
        title = doc_data["title"]
        domain = doc_data["domain"]
        sentences_list = []

        for s_idx, (text, is_anomaly, category, phrases) in enumerate(doc_data["sentences"]):
            spans = []
            for p in phrases:
                spans.append(create_span(text, p))

            sentences_list.append({
                "sentence_id": s_idx,
                "text": text,
                "is_anomaly": is_anomaly,
                "category": category,
                "token_spans": spans
            })

        documents.append({
            "doc_id": doc_id,
            "title": title,
            "domain": domain,
            "is_synthetic": True,
            "sentences": sentences_list
        })

    benchmark_dict = {
        "benchmark_metadata": {
            "dataset_name": "DGSA Multi-Domain Semantic Anomaly Benchmark v3.0",
            "is_synthetic": True,
            "description": "Gold-standard research benchmark featuring 20 diverse domain documents with curated semantic anomalies, verified character spans, and pure normal controls.",
            "disclaimer": "Curated for rigorous algorithmic verification and zero-leakage document-level baseline comparison.",
            "version": "3.0",
            "total_documents": len(documents)
        },
        "documents": documents
    }

    # Save JSON files
    json_path = Path("data/datasets/research_benchmark.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)

    synthetic_json = Path("data/datasets/synthetic_benchmark.json")
    with open(synthetic_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)

    # Flatten to CSV
    rows = []
    for doc in documents:
        d_id = doc["doc_id"]
        dom = doc["domain"]
        tit = doc["title"]
        for s in doc["sentences"]:
            rows.append({
                "doc_id": d_id,
                "title": tit,
                "domain": dom,
                "sentence_id": s["sentence_id"],
                "text": s["text"],
                "is_anomaly": s["is_anomaly"],
                "category": s["category"],
                "token_spans": json.dumps(s["token_spans"])
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/datasets/research_benchmark.csv", index=False)
    df.to_csv("data/datasets/synthetic_benchmark.csv", index=False)

    total_sentences = len(df)
    total_anomalies = int(df["is_anomaly"].sum())
    print(f"Generated 20 documents, {total_sentences} sentences, {total_anomalies} anomalies.")
    print("Category breakdown:")
    print(df[df["is_anomaly"] == 1]["category"].value_counts())


if __name__ == "__main__":
    build_and_save_dataset()
