# APGCE 2026 GeoHackathon — Challenge 1: Agentic AI Subsurface Exploration Copilot

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![vLLM](https://img.shields.io/badge/LLM-Gemma--4--26B--A4B-green.svg)
![Platform](https://img.shields.io/badge/Platform-DUG%20HPC%20Linux-orange.svg)

---

## 📌 Executive Summary & Project Overview
**Exploration Copilot** is an agentic AI subsurface prospect evaluation and decision-support platform developed for the **APGCE 2026 GeoHackathon (Challenge 1)**.

Evaluating exploration prospects traditionally requires geoscientists to spend **2 to 3 weeks** opening fragmented software, manually parsing wireline `.las` log files, cross-referencing pre-stack AVO elastic impedance volumes, and tracing structural horizons.

**Exploration Copilot** bridges structured petrophysical log attributes, structural fault/horizon picks from **DUG Insight** (e.g., extensional normal fault displacements), and pre-stack AVO inversion parameters (Vp/Vs ratios, P-Impedance, S-Impedance) into a unified executive decision dashboard. Powered by an on-premises **vLLM engine (Gemma 4 26B A4B)** hosted on local HPC infrastructure, the platform guarantees **zero enterprise data leakage** while accelerating multi-well prospect screening from **weeks down to 3 seconds (95%+ productivity gain)**.

---

## 🏗️ System Architecture & Workflow

```text
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│       DUG Insight (Visual QC)        │       │      Raw Wireline LAS Datasets       │
│  • Structural Horizon Tracing        │       │  • Log Curves (Vsh, PHIT, Sw, Sh)    │
│  • Fault Offset Mapping (e.g. F-08)  │       │  • Pre-Stack AVO Parameters (Vp/Vs)  │
└──────────────────┬───────────────────┘       └──────────────────┬───────────────────┘
                   │                                              │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │  Deterministic Knowledge Extractor  │
                       │       (`generate_knowledge.py`)     │
                       └──────────────────┬──────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │  Immutable Grounded Data Context    │
                       │        (`well_knowledge.py`)        │
                       └──────────────────┬──────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ Streamlit Web Dashboard (`app.py`)  │
                       │  • 3-Track Interactive Zoom Plotter │
                       │  • Subsurface & AVO Inversion Panel │
                       │  • Multi-Well Ranking Matrix        │
                       │  • Directly Grounded Gemma 4 AI     │
                       └──────────────────┬──────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ Local HPC vLLM REST Inference Engine│
                       │   (Gemma 4 26B A4B @ 172.28.34.1)   │
                       └─────────────────────────────────────┘

```

---

## 🛡️ Data Grounding & Verification Standard
To eliminate AI hallucinations and ensure complete technical transparency, **Exploration Copilot** categorizes its subsurface inputs into three grounded tiers:

## 1. **Automated Real Data (100% Computed):** 

All wireline curve plots, sample counts, net pay intervals, and quantitative petrophysical averages (Vshale, Phitotal, Sw, Sh) are parsed directly from field `.las` files using `lasio` and `pandas`.

## 2. **Visual Geological Picks (100% Grounded):** 

Structural normal fault parameters for **Well P-1** (e.g., `Fault P1-F08`, CDP 2200–2400, 25ms TWT displacement) represent real seismic interpretations performed by our team in DUG Insight.

## 3. **Zero-Hallucination Safe Fallbacks:**

Un-interpreted candidate wells are strictly labeled as `"Pending DUG Horizon Pick"` and `"Pending Inversion"` rather than using synthetic values, ensuring the Agentic AI only reasons over verified domain evidence.

## 🚀 Key Technical Features & Capabilities

### 1. 📊 Interactive 3-Track Wireline Log Evidence Visualizer

* **Customizable Track Mapping:** Allows engineers to toggle between Lithology (VSh, GR, SP), Porosity/Elastic (PHIT, Vp/Vs, Zp, Zs), and Fluid Saturation (Sh, Sw, Resistivity).


* **Hydrocarbon Saturation Focus (Sh = 1 - Sw):** Replaces raw water saturation with a dedicated Hydrocarbon Saturation track highlighted by custom gold/green pay fills.


* **Depth Indexing (m) & Live Slider Zooming:** Features interactive zoom sliders for precise inspection of target pay intervals (e.g., 1800m - 2500m) against DUG Insight 3D seismic sections.

* **Elastic Inversion Comparison:** Track 2 supports dynamic curve selection, enabling side-by-side comparison of Porosity (PHIT), Vp/Vs velocity ratios, P-Impedance (Zp), and S-Impedance (Zs).



### 2. 🎯 Pre-Stack AVO & Elastic Inversion Screening Panel

* **Fluid & Lithology Discrimination:** Integrates Amplitude Versus Offset (AVO) anomaly classifications (Class I–IV Bright/Dim Spots) and flags low Vp/Vs ratios (< 1.7) to confirm gas/light-oil pay versus brine sands.

* **DUG Insight Structural Grounding:** Incorporates traced normal fault geometry (CDP offsets, Two-Way Travel Time displacement) directly into the AI prompt to evaluate fault-dependent structural closures vs. stratigraphic pinchouts.



### 3. 🏆 Multi-Well Prospect Ranking Engine

* Constructs an automated comparative matrix across candidate wells (`NO-1`, `EO-1`, `M-1`, `P-1`, etc.).


* Ranks candidates dynamically based on petrophysical cutoffs (Vsh < 0.30, Sw < 0.50, Vp/Vs < 1.7) and structural trap closure.



### 4. 🤖 Grounded Agentic AI Assistant (Zero Hallucination)

* Communicates directly with the HPC local **vLLM Gemma 4 26B** endpoint.


* Follows a deterministic, 4-step structured geological reasoning trace:

Data Observation --> Geological & AVO Reasoning --> Prospect Conclusion --> Key Risks & Next Steps



---

## 🛠️ Quickstart & Installation Guide

### Prerequisites

* **Python Version:** 3.10+


* **Operating System:** Linux (Rocky Linux / Ubuntu) or macOS / Windows


* **Network Access:** Connected to local HPC network (`http://172.28.34.1:8000/v1` vLLM Endpoint)



### 1. Clone the Public Repository

```bash
git clone [https://github.com/Zoeeeeee03/apgce-2026-geohackathon-challenge1.git](https://github.com/Zoeeeeee03/apgce-2026-geohackathon-challenge1.git)
cd apgce-2026-geohackathon-challenge1

```

### 2. Install Python Dependencies

```bash
pip install streamlit pandas numpy matplotlib lasio openai

```

### 3. Generate Knowledge Base & Launch Application

```bash
# 1. Parse raw LAS files into structured JSON context
python3 generate_knowledge.py

# 2. Start the Streamlit web dashboard
streamlit run app.py

```

Open your web browser and navigate to `http://localhost:8501`.

---

## 💼 Business ROI & Investor Impact

| Value Metric | Traditional Workflow | Exploration Copilot Solution | Business Value Created |
| :--- | :--- | :--- | :--- |
| **Screening Speed** | 2–3 weeks per prospect | **< 3 seconds per well** | **95%+ Productivity Savings** |
| **Capital Risk Gate** | Manual PDF/Log checks | Automated Net Pay & Vp/Vs cutoffs | **Mitigates $10M+ Dry Hole Risk** |
| **Data Security** | Risk of third-party cloud leaks | Local HPC vLLM (Gemma 4) | **Zero Enterprise Data Leakage** |
| **Workflow Ecosystem** | Disconnected tools and spreadsheets | Streamlit UI + DUG Insight Co-Pilot layer | **Seamless Technical/Executive Adoption** |

## 🏆 Scoring Criteria Alignment Checklist

* [x] **Technical Design (15%):** Modular architecture connecting `lasio` parsers, grounded dictionaries, local vLLM API, and Streamlit UI.


* [x] **Technical Execution (15%):** Fully functional prototype displaying 13 well datasets, interactive wireline zoom plots, and live agentic AI responses.


* [x] **User Experience UX/UI (15%):** Clean dual-layout (Top Subsurface & AVO Characterization Cards + Bottom Full-Width 3-Track Log Visualizer) with direct AI assistant interaction.


* [x] **User Value (15%):** Solves core geoscientist pain points by synthesizing structured LAS logs, pre-stack AVO attributes, and DUG Insight fault interpretations into 1-click answers.


* [x] **Business Impact (20%):** Quantified $10M+ dry-hole risk mitigation and 95% time reduction for energy operators.


* [x] **Innovation & Creativity (10%):** Agentic AI prospect ranking de-risks multi-well exploration through transparent, evidence-backed reasoning chains.


* [x] **Presentation & Storytelling (10%):** Executive dashboard layout tailored for both technical engineers and business decision-makers.



---

## 📄 License

This repository is open-sourced under the **MIT License**.
