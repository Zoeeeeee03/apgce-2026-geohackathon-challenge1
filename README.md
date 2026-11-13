# 🌍 StrataMind AI — Subsurface Exploration Copilot (APGCE 2026 GeoHackathon — Challenge 1)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![vLLM](https://img.shields.io/badge/LLM-Gemma--4--26B--A4B-green.svg)

**An Agentic AI Subsurface Exploration Assistant that fuses 2D Seismic Volumes, Well-Tie Checkshots, 1D Wireline Petrophysics, and Analogue Field Knowledge to detect, validate, and explain hydrocarbon traps.**

---

## 🎯 1. What Problem Are We Solving?
Subsurface exploration teams spend **2 to 3 weeks** per prospect manually switching across fragmented software to correlate 2D seismic sections with wireline logs and historical analogue studies:
* **Seismic Ambiguity (False Positives):** 2D seismic sections reveal numerous structural and stratigraphic features—fault displacements, discontinuities, channels, onlaps, and erosional truncations—but seismic geometry alone cannot distinguish **true hydrocarbon-charged traps** from **brine-wet sands or acquisition noise**.
* **Siloed Data Interpretation:** Well logs (`.las`), seismic stack volumes (`.sgy`), time-depth checkshots (`.txt`), and analogue field literature live in disconnected workflows.
* **Lack of Explainability:** Conventional automated tools flag anomalies without explaining the step-by-step geological logic connecting trap geometry, reservoir quality, and top/lateral seal integrity.

---

## 👥 2. Who Are We Solving It For?
* **Exploration Geoscientists & Geophysicists:** Accelerating 2D seismic-to-well tie QC, structural vs. stratigraphic trap screening, and depth-specific petrophysical validation.
* **Subsurface Asset Teams & Decision-Makers:** De-risking prospect maturation with auditable, evidence-backed geological reasoning and analogue field comparisons before committing drilling capital.

---

## ⚙️ 3. How Does Our Solution Work?
**StrataMind AI** is an interactive, modular Streamlit exploration workbench powered by an on-premises **vLLM inference engine (`google/gemma-4-26B-A4B-it`)** running on local HPC infrastructure (`http://172.28.34.1:8000/v1`) for **zero enterprise data leakage**.

### End-to-End Subsurface Workflow
1. **1D Wireline Log Ingestion & 4-Track Visualizer (`.las`):**
   * Users upload any target well `.las` dataset to dynamically generate a **4-Track Wireline Log** with interactive depth zooming:
     * **Track 1 — Lithology (`VSH`):** Identifies clean reservoir sands (Vsh} < 0.30) vs. regional sealing shales.
     * **Track 2 — Storage Capacity (`PHIT`):** Highlights high-porosity reservoir intervals (Phi > 0.15).
     * **Track 3 — Hydrocarbon Saturation (Sh = 1 - Swt):** Isolates hydrocarbon-charged pay zones (Sh > 0.50) from brine-saturated intervals.
     * **Track 4 — Pay Summary Flags (`PS_GIP` / `PS_OIP`):** Displays active gas and oil pay indicators.
2. **2D Seismic Stack & Checkshot Well-Tie Viewer (`.sgy` + `checkshot.txt`):**
   * Users upload a 2D seismic stack volume (`.sgy`) to render an interactive HTML seismic cross-section directly in the interface.
   * Users can optionally upload a well checkshot file (`checkshot.txt`) to perform time-depth conversion and **overlay the well trajectory directly onto the 2D seismic stack**, aligning log depth intervals with seismic Two-Way Travel Time (TWT) reflectors.
3. **Cross-Domain Trap Detection & False-Positive Filtering (`StrataMind AI Chatbox`):**
   * **Step A — Seismic Geometry Screening:** StrataMind analyzes seismic discontinuities, fault offsets, channelized bodies, onlaps, and horizon truncations to nominate candidate structural or stratigraphic trap locations.
   * **Step B — Petrophysical & Seal Verification:** To eliminate brine sands and seismic noise, the AI cross-checks nominated seismic zones against the 1D well log curves—verifying reservoir porosity (`PHIT`), hydrocarbon saturation (`SH`), and overlying/adjacent shale seals (`VSH`).
   * **Step C — Explainable Reasoning & Analogue Grounding:** Using a structured RAG knowledge base (`resource_library.py` & `wellknowledge2.py`), StrataMind explains *why* a feature forms a valid structural or stratigraphic trap, compares it against known analogue fields, and recommends next exploration steps.

---

## 🏗️ System Architecture

```text
┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│   1D Well Log Upload (.las)   │   │  2D Seismic Stack (.sgy) &    │   │  Analogue & Domain Knowledge  │
│  • VSH, PHIT, SWT -> SH       │   │  Well Checkshot (.txt)        │   │  • Trap Diagnostic Rules      │
│  • Pay Flags (PS_GIP/PS_OIP)  │   │  • Interactive HTML Stack     │   │  • DUG Structural/AVO Picks   │
│  • 4-Track Log Plotter        │   │  • Time-Depth Well Overlay    │   │  • Analogue Field Library     │
└───────────────┬───────────────┘   └───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │                                   │
                └───────────────────────────────────┼───────────────────────────────────┘
                                                    ▼
                                 ┌─────────────────────────────────────┐
                                 │  StrataMind Multimodal Context &    │
                                 │  Seismic-Well Cross-Validator       │
                                 │  (`seismicprocessing.py` & loaders) │
                                 └──────────────────┬──────────────────┘
                                                    ▼
                                 ┌─────────────────────────────────────┐
                                 │   Local HPC vLLM Inference Engine   │
                                 │    (Gemma 4 26B A4B — Zero Leak)    │
                                 └──────────────────┬──────────────────┘
                                                    ▼
                                 ┌─────────────────────────────────────┐
                                 │   Streamlit Workbench (`idkkk.py`)  │
                                 │  • 1D Log + 2D Seismic Side-by-Side │
                                 │  • Trap Location Suggestions        │
                                 │  • Conversational Geo-Reasoning     │
                                 └─────────────────────────────────────┘

```

---

## 💡 4. What Makes It Innovative?

* **Seismic-to-Log Cross-Validation (Noise & Brine Filtering):** Instead of relying solely on seismic geometry—which often misclassifies water-wet sands or noise as prospects—StrataMind pairs 2D seismic features (faults, onlaps, truncations, channels) with 1D well-log petrophysics (`VSH`, `PHIT`, `SH`) to confirm both **trap closure** and **hydrocarbon charge**.
* **Interactive Checkshot Well-Tie Overlay:** Bridges the depth-vs-time gap directly in the browser by overlaying uploaded `checkshot.txt` well paths onto `.sgy` seismic sections.
* **Explainable 4-Step Geoscientific Reasoning:** Enforces transparent, auditable outputs:

1. Subsurface Observation --> 2. Trap Diagnosis (Structural vs. Stratigraphic) --> 3. Geological & Analogue Logic --> 4. Seal Risks & Next Steps


* **On-Premise Sovereign AI:** Powered by local HPC **Gemma 4 26B**, ensuring proprietary `.sgy` and `.las` datasets never leave the enterprise network.

---

## 📈 5. Value & Impact Created

* **95%+ Faster Prospect Evaluation:** Compresses multi-domain seismic-well correlation, trap screening, and analogue lookup from **2–3 weeks down to seconds**.
* **Mitigates Multi-Million-Dollar Dry-Hole Risk:** Filters out brine-bearing anomalies and breached fault compartments early by requiring simultaneous seismic trap geometry, seal presence, and hydrocarbon saturation validation.
* **Collaborative Human-AI Workflow:** Empowers geoscientists to interrogate specific seismic regions and depth intervals conversationally while retaining full visual control over raw logs and seismic sections.

---

## ✅ 6. Why Believe in Our Solution?

* **Live Working Prototype:** End-to-end Streamlit application processing real `.las` well logs, `.sgy` seismic stack volumes, `checkshot.txt` time-depth tables, and DUG Insight structural/AVO interpretations.
* **Hybrid Deterministic + Generative Design:** All log tracks, cutoffs, seismic cropping/rendering (`seismicprocessing.py`), and well-tie projections are computed deterministically in Python, while Gemma 4 synthesizes cross-domain geological explanations grounded in `resource_library.py` and `wellknowledge2.py`.
* **Built-In Scientific Integrity:** Explicitly distinguishes verified well/seismic observations from unmapped intervals so the AI never fabricates unverified subsurface structures.

---

## 📂 Project Structure

```text
├── idkkk.py                        # Main Streamlit application (1D well log, 2D seismic stack viewer, & StrataMind AI chatbox)
├── wellknowledge2.py             # Ground-truth well database, AVO attributes & DUG structural picks
├── interpretation_data_loader.py # Seismic (.sgy), checkshot (.txt), & subsurface data loaders
├── seismicprocessing.py          # Seismic cropping, HTML visualization & annotation utilities
├── resource_library.py           # Analogue field knowledge & geological reference library
└── README.md                     # Project documentation

```

---

## 🛠️ Quickstart & Installation Guide

### Prerequisites

* **Python:** 3.10+
* **Network:** Access to local HPC vLLM endpoint (`http://172.28.34.1:8000/v1`)

### 1. Clone the Repository

```bash
git clone [https://github.com/Zoeeeeee03/apgce-2026-geohackathon-challenge1.git](https://github.com/Zoeeeeee03/apgce-2026-geohackathon-challenge1.git)
cd apgce-2026-geohackathon-challenge1

```

### 2. Install Dependencies

```bash
pip install streamlit lasio segyio pandas numpy matplotlib plotly openai

```

### 3. Launch the Application

```bash
streamlit run idkkk.py

```

1. Open your browser at `http://localhost:8501`.
2. **Upload `.las` file** in the 1D Well Log panel to inspect the 4-track petrophysical logs.
3. **Upload `.sgy` stack volume** (and optional `checkshot.txt`) in the 2D Seismic panel to visualize the seismic section and well overlay.
4. **Query the StrataMind AI Chatbox** to detect candidate trap locations, differentiate structural vs. stratigraphic traps, and inspect step-by-step geological reasoning.

---

## 🏆 Challenge 1 Judging Alignment

* **Technical Design & Execution (30%):** Modular pipeline combining `lasio` log parsing, `segyio` seismic visualization, checkshot well-tie projection, analogue RAG grounding, and local vLLM inference.
* **User Experience UX/UI (15%):** Side-by-side 1D wireline log + 2D HTML seismic viewer paired with an interactive conversational copilot.
* **User Value & Business Impact (35%):** Directly solves false-positive seismic screening by cross-validating trap geometry with petrophysical seal and saturation data, cutting screening time by 95%+.
* **Innovation & Storytelling (20%):** Seamlessly connects seismic discontinuities/stratigraphic features with well-log fluid verification and analogue field reasoning.

---

## 📄 License

This project is open-sourced under the **MIT License**.

```

```
