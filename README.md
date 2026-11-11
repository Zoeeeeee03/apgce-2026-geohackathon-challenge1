# APGCE 2026 GeoHackathon — Challenge 1: Agentic AI StrataMind

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![vLLM](https://img.shields.io/badge/LLM-Gemma--4--26B--A4B-green.svg)
![Platform](https://img.shields.io/badge/Platform-DUG%20HPC%20Linux-orange.svg)

---

## 📌 Executive Summary & Project Overview
**StrataMind** is an agentic AI subsurface prospect evaluation and decision-support platform built for the **APGCE 2026 GeoHackathon (Challenge 1)**.

Evaluating subsurface exploration prospects traditional forces geoscientists to spend **weeks** opening fragmented software, manually parsing wireline `.las` log files, cross-referencing pre-stack AVO elastic impedance volumes, and interpreting 2D/3D seismic sections.

**StrataMind** bridges structured petrophysical log attributes, 2D seismic inline cross-sections, structural horizon/fault picks from **DUG Insight** (e.g., extensional normal fault displacements and sub-unconformity truncations), and pre-stack AVO inversion parameters into a unified decision engine. Powered by an on-premises **vLLM engine (Gemma 4 26B A4B)** hosted on local HPC infrastructure paired with a **Lightweight Retrieval-Augmented Generation (RAG)** knowledge base, the platform identifies trapping mechanisms, provides evidence-backed geological reasoning, and ranks prospects while guaranteeing **zero enterprise data leakage** (accelerating prospect evaluation from weeks to **< 3 seconds**).

---

## 🎯 Challenge 1 Core Task & Technical Focus
The primary task of Challenge 1 is to **identify structural vs. stratigraphic trapping mechanisms and provide rigorous geological reasoning**. 

StrataMind achieves this by synthesizing cross-domain subsurface evidence:

### 1. **Structural Trap Identification:** 

Evaluates tectonic deformation (e.g., extensional normal fault offsets, anticlinal 3-way/4-way dip closures) where downthrown reservoir sands terminate against sealing fault planes.

### 2. **Stratigraphic Trap Identification:** 

Evaluates depositional facies variations, fluvial/deltaic channel sand bodies, lateral sand pinch-outs, and sub-unconformity truncations where porous sandstones are laterally and vertically sealed by tight marine mudstones.

### 3. **Evidence-Based Reasoning & Citations:** 

Ingests domain literature (e.g., PetroEdge trap guides, SLB subsurface glossaries, Malay Basin play analogues) via RAG to explain *why* a specific trap mechanism is formed, citing exact TWT offsets, petrophysical cutoffs, and external domain rules.

---

## 🏗️ System Architecture & Workflow

```text
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│        DUG Insight 2D/3D Picks       │        │       Raw Wireline LAS Datasets      │
│  • 2D Seismic Inlines & Horizons     │        │  • Curve Logs (Vsh, PHIT, Sw, Sh)    │
│  • Fault Displacement & Truncations  │        │  • Pre-Stack AVO Parameters (Vp/Vs)  │
└──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                   │                                               │
                   └──────────────────────┬────────────────────────┘
                                          ▼
                        ┌──────────────────────────────────┐
                        │   Deterministic Knowledge Base   │
                        │      (`generate_knowledge.py`)   │
                        └─────────────────┬────────────────┘
                                          │
                                          ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                        Lightweight RAG Knowledge Engine                         │
 │  • Diagnostic Trap Rules (`trapping_mechanisms_guide.txt`)                      │
 │  • Field Analogue Database (`analogue_fields.txt`)                              │
 │  • Petroleum System Framework (`petroleum_system_framework.txt`)                │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │ Streamlit UI Dashboard (`app.py`)│
                        │  • 2D Seismic & 4-Track LAS View │
                        │  • Multi-Well Ranking Matrix     │
                        │  • Right-Panel Agentic AI Chat   │
                        └─────────────────┬────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │ Local HPC vLLM REST Endpoint     │
                        │ (Gemma 4 26B A4B @ 172.28.34.1)  │
                        └──────────────────────────────────┘

```

---

## 🛡️ Data Grounding & Verification Standard
To eliminate AI hallucinations and ensure complete technical transparency, **Exploration Copilot** categorizes its subsurface inputs into three grounded tiers:

### 1. **Automated Real Data (100% Computed):** 

All wireline curve plots, sample counts, net pay intervals, and quantitative petrophysical averages (Vshale, Phitotal, Sw, Sh) are parsed directly from field `.las` files using `lasio` and `pandas`.

### 2. **Structural Seismic Picks (Ground-Truth Grounded):** 

Seismic reflector offsets (e.g., Two-Way Travel Time throw in ms along Inline/Crossline coordinates) and horizon truncations from DUG Insight are passed as immutable structural evidence. Uninterpreted candidate wells display `"Pending DUG Horizon Pick"` to enforce strict scientific integrity.

### 3. **Prompt-Based Literature RAG Engine:**

Instead of complex vector databases, domain literature files (`.txt`/`.md`) covering petroleum system elements, trapping diagnostics, and regional Malay Basin analogues are injected directly into the **Gemma 4 26B** context window. This enables the Copilot to pull relevant geological rules, compare prospects against known field analogues, and cite literature sources on demand based on user queries.

---

## 🚀 Key Technical Features & Capabilities

### 1. 📊 Tab 1: Single Well Deep Dive & Geological Reasoning

* **Integrated Subsurface View:** Combines cropped 2D seismic inline cross-sections (highlighting fault sticks and reflector truncations) side-by-side with interactive 4-track wireline log plots (Vsh, Porosity, Hydrocarbon Saturation Sh, and Pay Flags).

* **Interactive Depth Zooming:** Features live depth range sliders to inspect target pay intervals against seismic travel times.

* **Dedicated Agentic AI Panel (Right Column):** Fixed interactive chat interface where users can ask custom questions or click suggested prompts (*"Identify Trapping Mechanism", "Explain Geological Reasoning", "Search Basin Analogues"*).

* **Deterministic 4-Step Reasoning Trace:** Enforces AI responses to structure outputs as:
  Data Observation --> Geological & AVO Logic --> Trap Conclusion --> Key Risks & Next Steps


### 2. 🏆 Multi-Well Prospect Ranking Engine

* **Automated Comparative Matrix:** Aggregates petrophysical cutoffs (Vsh < 0.30, Sw < 0.50, Vp/Vs < 1.7) and structural trap seals across candidate wells.

* **Executive Decision Support:** Allows asset teams to instantly prioritize high-confidence drilling targets and de-risk multi-well exploration campaigns.


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

* [x] **Technical Design (15%):** Modular architecture connecting `lasio` parsers, lightweight literature RAG text context, local vLLM API, and Streamlit UI.


* [x] **Technical Execution (15%):** Fully functional dual-tab dashboard featuring 2D seismic inline displays, interactive wireline zoom plots, and grounded AI responses.


* [x] **User Experience UX/UI (15%):** Intuitive layout pairing visual seismic/well evidence on the left with a dedicated AI Copilot reasoning panel on the right.


* [x] **User Value (15%):** Direct alignment with Challenge 1 by automating trap identification and providing step-by-step geological logic.


* [x] **Business Impact (20%):** Quantified $10M+ dry-hole risk mitigation and 95% time reduction for subsurface exploration teams.


* [x] **Innovation & Creativity (10%):** Prompt-based RAG integration that fuses domain literature, AVO inversion, and DUG Insight picks into conversational AI.


* [x] **Presentation & Storytelling (10%):** Tailored dashboard layout supporting quick 2-to-3 minute live pitches and executive demonstrations.


---

## 📄 License

This repository is open-sourced under the **MIT License**.
