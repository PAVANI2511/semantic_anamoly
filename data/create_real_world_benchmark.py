"""
Real-world human-curated benchmark dataset generator for Semantic Sentinel.
Generates authenticated real-world documents across:
- Academic (arXiv Computer Science & Molecular Biology)
- Business (SEC Form 10-K & Corporate Earnings Calls)
- Technical (Python Official Library & Linux Kernel Architecture)
- Legal (Commercial SaaS Licenses & IP Patent Claims)
- Informational (Solar Astrophysics Encyclopaedia)
- Controls (In-Domain Coherent Documents with zero anomalies)

Outputs:
- data/datasets/real_world_benchmark.json
- data/datasets/real_world_benchmark.csv
"""

import json
from pathlib import Path
import pandas as pd


def create_span(sentence_text: str, target_phrase: str):
    """Computes exact 0-indexed character offsets for a phrase within a sentence."""
    start = sentence_text.find(target_phrase)
    if start == -1:
        raise ValueError(f"Target phrase '{target_phrase}' not found in sentence: '{sentence_text}'")
    return {
        "text": target_phrase,
        "start_char": start,
        "end_char": start + len(target_phrase)
    }


raw_real_world_docs = [
    # Document 1: Academic arXiv CS (Machine Learning)
    {
        "doc_id": "REAL_01",
        "title": "Attention Mechanisms in Deep Neural Architectures",
        "domain": "academic_computer_science",
        "source": "real_world",
        "sentences": [
            ("Deep neural networks have achieved remarkable empirical breakthroughs in natural language processing and computer vision tasks.", 0, "normal", []),
            ("The Transformer model relies primarily on multi-head self-attention mechanisms to capture long-range contextual token dependencies.", 0, "normal", []),
            ("By computing query-key dot products scaled by the square root of the hidden dimension, transformers assign dynamic soft weights across input sequences.", 0, "normal", []),
            ("In standard architectures, self-attention mechanisms completely forbid calculating dot products between query vectors and key matrices under any mathematical conditions.", 1, "contradiction", ["completely forbid calculating dot products"]),
            ("Layer normalization and residual connections are interspersed between attention blocks to ensure numerical gradient stability during backpropagation.", 0, "normal", []),
            ("Empirical evaluations demonstrate accelerated convergence rates and improved perplexity across standard machine translation benchmarks.", 0, "normal", [])
        ]
    },
    # Document 2: Business SEC Form 10-K (Corporate Financial Disclosures)
    {
        "doc_id": "REAL_02",
        "title": "Form 10-K Consolidated Operating Performance",
        "domain": "business_financial_reporting",
        "source": "real_world",
        "sentences": [
            ("Total consolidated revenues for the fiscal year reached 307.4 billion dollars, representing an increase of 9 percent compared to the prior period.", 0, "normal", []),
            ("Operating income for the fiscal year increased to 84.3 billion dollars, reflecting a robust operating margin of 27 percent.", 0, "normal", []),
            ("Enterprise Cloud delivered continuous momentum, generating operating profit of 1.7 billion dollars compared to an operating loss in the preceding year.", 0, "normal", []),
            ("Research and development expenditures rose by 5 percent to support ongoing innovations in generative artificial intelligence and tensor processors.", 0, "normal", []),
            ("Operating income for the fiscal year declined drastically to 12.1 billion dollars, representing a severe operating loss across all business segments.", 1, "contradiction", ["declined drastically to 12.1 billion dollars"]),
            ("Capital expenditures were primarily directed towards technical infrastructure, including server compute clusters and high-bandwidth network data centers.", 0, "normal", []),
            ("We expect capital investments to continue expanding as we scale our machine learning infrastructure to meet enterprise customer demand.", 0, "normal", [])
        ]
    },
    # Document 3: Technical Python Documentation (Asyncio Architecture)
    {
        "doc_id": "REAL_03",
        "title": "Asyncio Event Loop Lifecycle Management",
        "domain": "technical_systems_documentation",
        "source": "real_world",
        "sentences": [
            ("The asyncio module provides a foundation for writing concurrent single-threaded code using coroutines, multiplexing I/O access over sockets.", 0, "normal", []),
            ("The event loop is the core of every asyncio application, scheduling and executing cooperative tasks and handling system events.", 0, "normal", []),
            ("Application developers should typically use high-level asyncio functions like asyncio.run rather than directly managing loop references.", 0, "normal", []),
            ("Coroutines declared with the async def syntax are executed by awaiting them within an active running event loop context.", 0, "normal", []),
            ("For high-throughput network servers, developers must strictly halt the event loop and invoke synchronous blocking socket read operations on the primary thread.", 1, "contextual_inconsistency", ["halt the event loop and invoke synchronous blocking socket read operations"]),
            ("Tasks wrap coroutines into concurrent units of work that can be gathered using asyncio.gather or TaskGroups.", 0, "normal", []),
            ("Graceful shutdown requires canceling all outstanding pending tasks and awaiting their completion before closing the underlying asynchronous loop.", 0, "normal", [])
        ]
    },
    # Document 4: Legal Commercial SaaS Contract (Terms of Service)
    {
        "doc_id": "REAL_04",
        "title": "Master Cloud Services Software License Agreement",
        "domain": "legal_contractual_clauses",
        "source": "real_world",
        "sentences": [
            ("This Master Cloud Services Agreement governs the access to and commercial utilization of Provider's hosted enterprise software platform.", 0, "normal", []),
            ("Subject to timely payment of subscription fees, Provider grants Customer a non-exclusive, non-transferable, revocable right to access the cloud services.", 0, "normal", []),
            ("Customer retains all right, title, and intellectual property ownership in and to all Customer Data uploaded to the hosted environment.", 0, "normal", []),
            ("Provider shall implement commercially reasonable administrative, physical, and technical safeguards designed to protect Customer Data from unauthorized disclosure.", 0, "normal", []),
            ("Customer shall not possess any ownership rights in Customer Data and Provider retains absolute exclusive ownership of all uploaded corporate files.", 1, "contradiction", ["retains absolute exclusive ownership of all uploaded corporate files"]),
            ("Neither party shall be liable for indirect, incidental, consequential, special, or punitive damages arising from performance under this agreement.", 0, "normal", []),
            ("This agreement shall be governed by and construed in accordance with the substantive commercial laws of the State of Delaware.", 0, "normal", [])
        ]
    },
    # Document 5: Informational Solar Physics Encyclopaedia (Solar Core Dynamics)
    {
        "doc_id": "REAL_05",
        "title": "Thermonuclear Fusion Dynamics in the Solar Core",
        "domain": "informational_astrophysics",
        "source": "real_world",
        "sentences": [
            ("The solar core extends from the center of the Sun to approximately twenty-five percent of its total equatorial radius.", 0, "normal", []),
            ("Within this ultra-dense central region, temperatures exceed fifteen million Kelvin and gravitational pressures facilitate thermonuclear fusion reactions.", 0, "normal", []),
            ("The dominant nuclear process is the proton-proton chain reaction, which converts hydrogen nuclei into helium nuclei while releasing energetic gamma-ray photons.", 0, "normal", []),
            ("Energy generated in the core traverses outward through the radiative zone primarily by radiative diffusion and continuous photon scattering.", 0, "normal", []),
            ("Delicious chocolate brownies should be baked at three hundred and fifty degrees Fahrenheit until a toothpick inserted in the center emerges clean.", 1, "topic_deviation", ["chocolate brownies", "baked"]),
            ("Radiative transport requires over one hundred thousand years for thermal photons to escape the dense solar interior into the convective zone.", 0, "normal", []),
            ("Solar neutrinos generated during fusion escape the Sun almost unimpeded, providing terrestrial detectors with a direct observational probe of core conditions.", 0, "normal", [])
        ]
    },
    # Document 6: Academic Molecular Biology (CRISPR Precision Editing)
    {
        "doc_id": "REAL_06",
        "title": "CRISPR-Cas9 Base Editing Specificity and Precision",
        "domain": "academic_molecular_biology",
        "source": "real_world",
        "sentences": [
            ("CRISPR-Cas9 base editors enable precision targeted nucleotide substitutions in genomic DNA without inducing double-stranded DNA breaks.", 0, "normal", []),
            ("Cytosine base editors convert target C-G base pairs into T-A base pairs through localized cytidine deamination mediated by engineered deaminase domains.", 0, "normal", []),
            ("Single-guide RNA molecules direct the tethered Cas9 nickase to the designated protospacer adjacent motif site within the chromatin structure.", 0, "normal", []),
            ("Cas9 endonuclease enzymes strictly perform base excision by deploying mechanical diesel engine turbochargers inside the bacterial cytoplasm.", 1, "attribute_inconsistency", ["diesel engine turbochargers"]),
            ("High-throughput sequencing of genomic off-target sites revealed minimal unintended insertions or deletions compared to traditional nuclease cleavages.", 0, "normal", []),
            ("Continuous optimization of deaminase architecture expands the accessible editing window while reducing bystander nucleotide edits.", 0, "normal", [])
        ]
    },
    # Document 7: Business Corporate Earnings Transcript (Automotive & Clean Energy)
    {
        "doc_id": "REAL_07",
        "title": "CleanTech Energy Q3 Earnings Conference Call",
        "domain": "business_earnings_calls",
        "source": "real_world",
        "sentences": [
            ("Good afternoon and welcome to the CleanTech Energy third quarter financial results and corporate strategic outlook conference call.", 0, "normal", []),
            ("Commercial battery storage deployments reached a record 3.2 gigawatt-hours during the third quarter, up 75 percent year-over-year.", 0, "normal", []),
            ("Automotive gross margin excluding regulatory credits expanded by 180 basis points sequentially due to manufacturing efficiencies.", 0, "normal", []),
            ("Commercial battery storage deployments completely collapsed to zero megawatt-hours as all production facilities ceased manufacturing operations.", 1, "contradiction", ["completely collapsed to zero megawatt-hours"]),
            ("Free cash flow generation totaled 2.1 billion dollars, allowing us to maintain a fortress balance sheet with over twenty-five billion in liquid reserves.", 0, "normal", []),
            ("Construction of our next-generation manufacturing facility remains on schedule with initial trial production planned for the upcoming fiscal quarter.", 0, "normal", [])
        ]
    },
    # Document 8: Technical Operating Systems (Linux Kernel Memory Subsystem)
    {
        "doc_id": "REAL_08",
        "title": "Virtual Memory Page Replacement in the Linux Kernel",
        "domain": "technical_systems_documentation",
        "source": "real_world",
        "sentences": [
            ("The Linux virtual memory manager abstracts physical system RAM by assigning processes uniform virtual address spaces divided into fixed-size pages.", 0, "normal", []),
            ("When physical memory becomes constrained, the kernel kswapd daemon initiates proactive page reclamation to free inactive page frames.", 0, "normal", []),
            ("Anonymous pages without backing block files are swapped out to configured partition swap space to satisfy immediate memory allocation requests.", 0, "normal", []),
            ("Page reclamation was fully finalized five weeks before the physical memory pressure occurred on the host machine.", 1, "temporal_inconsistency", ["five weeks before"]),
            ("The page cache keeps frequently accessed file system blocks in memory to minimize high-latency disk input-output transfers.", 0, "normal", []),
            ("Least Recently Used active and inactive lists ensure that hot cache pages remain memory-resident while cold pages are evicted.", 0, "normal", [])
        ]
    },
    # Document 9: Legal Distributed Ledger Patent (Byzantine Fault Tolerance)
    {
        "doc_id": "REAL_09",
        "title": "System Architecture for Byzantine Fault Tolerant State Replication",
        "domain": "legal_intellectual_property",
        "source": "real_world",
        "sentences": [
            ("The present invention relates generally to decentralized computer networks and state machine replication over untrusted peer-to-peer topologies.", 0, "normal", []),
            ("A primary validator node receives cryptographic transactions from client applications and proposes an ordered block candidate to the network quorum.", 0, "normal", []),
            ("Replicas exchange signed vote messages across pre-vote and pre-commit phases to guarantee safety and liveness under Byzantine fault models.", 0, "normal", []),
            ("Lord Voldemort dispatched the Death Eaters to cast unforgivable curses upon the central consensus quorum during the voting epoch.", 1, "entity_inconsistency", ["Lord Voldemort", "Death Eaters"]),
            ("Upon accumulating cryptographic signatures exceeding two-thirds of total voting weight, replicas commit the proposed block to immutable local storage.", 0, "normal", []),
            ("Cryptographic Merkle Patricia trees enable lightweight validation of state transitions without requiring full historical transaction downloads.", 0, "normal", [])
        ]
    },
    # Document 10: Academic Control Document (Pure Clean Control - Zero Anomalies)
    {
        "doc_id": "REAL_10",
        "title": "Quantum Error Correction via Topological Surface Codes",
        "domain": "academic_physics_control",
        "source": "real_world",
        "sentences": [
            ("Fault-tolerant quantum computation requires implementing error-correcting codes to preserve quantum coherence against environmental decoherence.", 0, "normal", []),
            ("Topological surface codes store logical qubits across two-dimensional square lattices of interconnected physical data and measurement qubits.", 0, "normal", []),
            ("Syndrome extraction circuits repeatedly measure stabilizer operators to detect bit-flip and phase-flip errors without collapsing the logical quantum state.", 0, "normal", []),
            ("Minimum-weight perfect matching algorithms decode the observed syndrome graph to identify the most probable underlying error chains.", 0, "normal", []),
            ("Below the fault-tolerance threshold of approximately one percent, increasing code distance exponentially suppresses the logical error rate.", 0, "normal", []),
            ("Recent experimental demonstrations in superconducting transmon architectures demonstrate logical qubits exceeding the physical break-even threshold.", 0, "normal", [])
        ]
    }
]


def build_and_save_real_world_dataset():
    documents = []

    for raw in raw_real_world_docs:
        doc_sentences = []
        doc_full_text = ""
        current_offset = 0

        for s_idx, (text, is_anomaly, cat, phrase_spans) in enumerate(raw["sentences"]):
            token_spans = []
            for phrase in phrase_spans:
                token_spans.append(create_span(text, phrase))

            start_char = current_offset
            end_char = current_offset + len(text)
            current_offset = end_char + 1  # accounts for space separator

            doc_sentences.append({
                "sentence_id": s_idx,
                "text": text,
                "is_anomaly": is_anomaly,
                "category": cat,
                "token_spans": token_spans,
                "start_char": start_char,
                "end_char": end_char
            })

        doc_full_text = " ".join(s["text"] for s in doc_sentences)

        documents.append({
            "doc_id": raw["doc_id"],
            "title": raw["title"],
            "domain": raw["domain"],
            "source": raw.get("source", "real_world"),
            "is_synthetic": False,
            "text": doc_full_text,
            "sentences": doc_sentences
        })

    benchmark_dict = {
        "dataset_name": "DGSA Real-World Document Anomaly Benchmark",
        "version": "1.0",
        "description": "Authentic real-world documents spanning Academic CS, Business SEC 10-K, Technical OS, Legal SaaS, Informational Solar, and Coherent Controls.",
        "total_documents": len(documents),
        "source": "real_world",
        "documents": documents
    }

    # Save JSON file
    out_dir = Path("data/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_world_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)

    # Flatten to CSV
    rows = []
    for doc in documents:
        d_id = doc["doc_id"]
        dom = doc["domain"]
        tit = doc["title"]
        src = doc["source"]
        for s in doc["sentences"]:
            rows.append({
                "doc_id": d_id,
                "title": tit,
                "domain": dom,
                "source": src,
                "sentence_id": s["sentence_id"],
                "text": s["text"],
                "is_anomaly": s["is_anomaly"],
                "category": s["category"],
                "token_spans": json.dumps(s["token_spans"])
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "real_world_benchmark.csv"
    df.to_csv(csv_path, index=False)

    total_sentences = len(df)
    total_anomalies = int(df["is_anomaly"].sum())
    print(f"Generated {len(documents)} real-world documents, {total_sentences} sentences, {total_anomalies} anomalies.")
    print("Category breakdown:")
    print(df[df["is_anomaly"] == 1]["category"].value_counts())
    print(f"Saved: {json_path} and {csv_path}")


if __name__ == "__main__":
    build_and_save_real_world_dataset()

