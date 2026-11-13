from resource_library import format_resource_context
from interpretation_engine import analyse_seismic_image
from interpretation_data_loader import load_interpretation_dataset

import streamlit.components.v1 as components
from html import escape

import os
import io
import json
import base64
import mimetypes
import uuid
import lasio
import streamlit as st

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wellknowledge2 import (
    analyse_segy_file,
    get_copilot_response
)

from seismicprocessing import (
    read_segy_preview,
    plot_seismic_section,
    get_seismic_summary,
    read_checkshot_txt,
    add_well_correlation_overlay
)

import tempfile
import textwrap
import requests
from PIL import Image

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StrataMind | Subsurface Exploration Copilot",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GEOLOGICAL INTERPRETATION DATA
# ============================================================

@st.cache_data(show_spinner="Loading well horizons and fault data...")
def get_interpretation_data():
    return load_interpretation_dataset()


def prepare_well_geological_context(dataset, well_name):
    """
    Prepare a concise geological reference from the selected well.

    Sends horizon and fault information to Gemma as supporting
    evidence without overwhelming the model with every data point.
    """

    horizons = dataset["horizons"].get(
        well_name, pd.DataFrame()
    )

    faults = dataset["faults"].get(
        well_name, pd.DataFrame()
    )

    context = []

    context.append(f"SELECTED REFERENCE WELL: {well_name}")

    # --------------------------------------------------------
    # HORIZON INFORMATION
    # --------------------------------------------------------

    context.append("\nHORIZON PICK DATA:")

    if horizons.empty:
        context.append("No horizon data available.")
    else:
        context.append(
            f"Number of horizon files: "
            f"{horizons['SourceFile'].nunique()}"
        )

        context.append(
            f"Total horizon points: {len(horizons)}"
        )

        context.append(
            "Columns: IL, CL, TWT_ms"
        )

        for horizon_id, group in horizons.groupby("HorizonID"):

            context.append(
                f"\nHorizon {horizon_id}: "
                f"{group['SourceFile'].iloc[0]}"
            )

            context.append(
                f"Point count: {len(group)}"
            )

            context.append(
                f"IL range: {group['IL'].min():.2f} "
                f"to {group['IL'].max():.2f}"
            )

            context.append(
                f"CL range: {group['CL'].min():.2f} "
                f"to {group['CL'].max():.2f}"
            )

            context.append(
                f"TWT range: {group['TWT_ms'].min():.2f} "
                f"to {group['TWT_ms'].max():.2f} ms"
            )

            # Provide representative picks, not every point.
            sample = group[
                ["IL", "CL", "TWT_ms"]
            ].iloc[
                sorted(set([
                    0,
                    len(group) // 4,
                    len(group) // 2,
                    3 * len(group) // 4,
                    len(group) - 1
                ]))
            ]

            context.append(
                "Representative picks (IL, CL, TWT_ms):"
            )

            context.append(
                sample.round(2).to_string(index=False)
            )

    # --------------------------------------------------------
    # FAULT INFORMATION
    # --------------------------------------------------------

    context.append("\nFAULT PICK DATA:")

    if faults.empty:
        context.append("No fault data available.")
    else:
        context.append(
            f"Number of fault files: "
            f"{faults['SourceFile'].nunique()}"
        )

        context.append(
            f"Total fault points: {len(faults)}"
        )

        context.append(
            "Columns: FaultID, StickID, IL, CL, TWT_ms"
        )

        for source_file, file_group in faults.groupby("SourceFile"):

            context.append(f"\nFault file: {source_file}")

            for fault_id, group in file_group.groupby("FaultID"):

                context.append(
                    f"Fault ID {fault_id}: "
                    f"{len(group)} points"
                )

                sample = group[
                    ["IL", "CL", "TWT_ms"]
                ].head(10)

                context.append(
                    sample.round(2).to_string(index=False)
                )

    context.append(
        "\nIMPORTANT DATA LIMITATIONS:"
        "\n- These are interpreted horizon and fault picks."
        "\n- They are reference data, not automatic proof of a trap."
        "\n- Do not assume the picks correspond to the displayed "
        "seismic section unless their coordinates are verified."
        "\n- Do not invent missing geological relationships."
        "\n- Distinguish supplied picks from features observed "
        "directly in the seismic image."
    )

    return "\n".join(context)

# ============================================================
# 2. GEMMA MODEL CONFIGURATION
# ============================================================

GEMMA_URL = os.getenv(
    "GEOH_VLLM_URL",
    "http://172.28.34.1:8000/v1"
)

GEMMA_MODEL = os.getenv(
    "GEOH_MODEL",
    "google/gemma-4-26B-A4B-it"
)

GEMMA_API_KEY = os.getenv(
    "GEOH_API_KEY",
    "token-not-needed"
)

GEMMA_CHAT_URL = (
    f"{GEMMA_URL.rstrip('/')}/chat/completions"
)

# ============================================================
# 3. GEOLOGICAL SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are StrataMind, an AI assistant specialised in seismic
interpretation and petroleum geology.

Your purpose is to help users interpret seismic sections,
well data, horizons, amplitude information and geological
structures.

You should assist with the identification and interpretation
of the following features:

STRUCTURAL FEATURES:
- Normal faults
- Reverse faults
- Fault displacement
- Anticlines and synclines
- Structural highs
- Fault closures
- Four-way closures
- Three-way closures

STRATIGRAPHIC FEATURES:
- Onlap
- Downlap
- Toplap
- Unconformities
- Pinch-outs
- Channels
- Reefs
- Stratigraphic traps

SEISMIC AMPLITUDE FEATURES:
- Bright spots
- Dim spots
- Flat spots
- Amplitude anomalies
- Chaotic reflections
- Reflection continuity
- Seismic facies changes

INTERPRETATION RULES:

1. Distinguish direct observations from geological interpretations.

2. Never invent faults, horizons, well information or coordinates.

3. A bright spot alone does not prove the presence of hydrocarbons.

4. Explain the seismic evidence supporting each interpretation.

5. Consider alternative geological explanations when appropriate.

6. Do not claim that a structural closure is proven using
   only one 2D seismic section.

7. If the uploaded data is insufficient, clearly state
   what additional information is required.

8. Use appropriate geological terminology.

9. Explain interpretations in a clear and understandable
   manner suitable for geology students.

10. When analysing an image, refer to visible seismic
    reflection patterns and their relative positions.
"""


# ============================================================
# 4. APPLICATION DESIGN
# ============================================================


st.markdown(
    """
    <style>

    .stApp {
        background-color: #F4F3E9;
        color: #303F32;
    }

    [data-testid="stSidebar"] {
        background-color: #E5E8D8;
    }

    .block-container {
        max-width: 100%;
        padding-top: 1.8rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 750;
        color: #304C36;
        margin-top: 0.3rem;
        margin-bottom: 6px;
        text-align: center;
    }

    .hero-title span {
        color: #81945B;
    }

    .hero-subtitle {
        color: #777D70;
        font-size: 14px;
        margin-top: 0;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Main workspace panel borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid #405A38 !important;
        border-radius: 12px;
    }

    /* Chat message appearance */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #DDE0D1;
        background-color: #F8F8F2;
        padding: 10px;
        max-width: 95%;
    }

    /* Keep user messages on the right */
    div[data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        margin-left: auto;
    }

    /* Keep AI messages on the left */
    div[data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        margin-right: auto;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        border: 1px solid #81945B;
    }

    /* Suggested question buttons */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #E8ECDD;
        color: #354D35;
        font-size: 12px;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #D6DFC8;
        border-color: #526B45;
        color: #304C36;
    }

    div[data-testid="stForm"] {
        background-color: #E9EBDD;
        border: 1px solid #D3D9C6;
        padding: 10px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 5. SESSION STATE
# ============================================================

if "conversations" not in st.session_state:

    first_id = str(uuid.uuid4())

    st.session_state.conversations = {
        first_id: {
            "title": "New conversation",
            "messages": []
        }
    }

    st.session_state.active_conversation = first_id


if "active_conversation" not in st.session_state:

    first_id = list(st.session_state.conversations.keys())[0]

    st.session_state.active_conversation = first_id


def current_conversation():

    return st.session_state.conversations[
        st.session_state.active_conversation
    ]


def create_new_conversation():

    new_id = str(uuid.uuid4())

    st.session_state.conversations[new_id] = {
        "title": "New conversation",
        "messages": []
    }

    st.session_state.active_conversation = new_id


# Store uploaded well-log data
if "well_log_df" not in st.session_state:
    st.session_state.well_log_df = None

if "well_log_filename" not in st.session_state:
    st.session_state.well_log_filename = None

if "seismic_result" not in st.session_state:
    st.session_state.seismic_result = None

# Interactive 2D seismic data
if "seismic_section" not in st.session_state:
    st.session_state.seismic_section = None

if "seismic_filename" not in st.session_state:
    st.session_state.seismic_filename = None

if "selected_seismic_stack" not in st.session_state:
    st.session_state.selected_seismic_stack = "Full Stack"

if "detected_curves" not in st.session_state:
    st.session_state.detected_curves = {}

if "pay_intervals" not in st.session_state:
    st.session_state.pay_intervals = []

if "petro_status" not in st.session_state:
    st.session_state.petro_status = []

if "well_seismic_correlation" not in st.session_state:
    st.session_state.well_seismic_correlation = None

if "selected_seismic_x_range" not in st.session_state:
    st.session_state.selected_seismic_x_range = None

if "selected_seismic_y_range" not in st.session_state:
    st.session_state.selected_seismic_y_range = None

if "seismic_interpretation_result" not in st.session_state:
    st.session_state.seismic_interpretation_result = None

# ============================================================
# AUTOMATIC WELL-LOG CURVE DETECTION
# ============================================================

CURVE_ALIASES = {
    "DEPTH": ["DEPTH", "DEPT", "MD"],
    "GR": ["GR", "GAMMA", "GAM", "SGR"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENS"],
    "NPHI": ["NPHI", "TNPH", "NEUT"],
    "DT": ["DT", "DTC", "AC"],
    "RT": [
        "RT", "ILD", "LLD", "LLS", "ILM",
        "LLM", "MSFL", "RDEP", "RES", "RESD"
    ],
    "VSH": ["VSH", "VCL", "VCLAY"],
    "PHIT": ["PHIT", "PHI", "POR"],
    "SW": ["SW", "SWE", "SWT", "SW_ARCHIE"],
    "SH": ["SH", "SHC", "SH_HC"],
    "PAY": ["PAY", "PAY_FLAG", "NET_PAY_FLAG"]
}

def detect_well_log_curves(df):

    detected = {}

    columns = {
        str(col).strip().upper(): col
        for col in df.columns
    }

    for standard_name, aliases in CURVE_ALIASES.items():

        for alias in aliases:

            if alias in columns:
                detected[standard_name] = columns[alias]
                break

    return detected

# ============================================================
# PETROPHYSICAL CALCULATION ENGINE
# ============================================================

def calculate_petrophysical_curves(df, detected, params):

    df = df.copy()
    detected = detected.copy()

    status = []

    # --------------------------------------------------------
    # Helper: retrieve and convert a curve
    # --------------------------------------------------------

    def get_curve(curve_name):

        column = detected.get(curve_name)

        if column and column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            return values

        return None

    # --------------------------------------------------------
    # Helper: convert percentage values into fractions
    # --------------------------------------------------------

    def to_fraction(values):

        if values is None:
            return None

        values = values.copy()

        valid = values.dropna()

        if len(valid) > 0 and valid.max() > 1.5:

            if valid.max() <= 100:
                values = values / 100

        return values

    # --------------------------------------------------------
    # 1. VSH: use existing curve or calculate from GR
    # --------------------------------------------------------

    vsh = get_curve("VSH")

    if vsh is not None:

        vsh = to_fraction(vsh)
        status.append("VSH: Existing LAS curve")

    else:

        gr = get_curve("GR")

        if gr is not None and gr.notna().any():

            gr_min = gr.quantile(0.05)
            gr_max = gr.quantile(0.95)

            if gr_max > gr_min:

                vsh = (
                    (gr - gr_min) /
                    (gr_max - gr_min)
                ).clip(0, 1)

                df["VSH_CALC"] = vsh
                detected["VSH"] = "VSH_CALC"

                status.append(
                    "VSH: Calculated using linear GR method"
                )

            else:

                status.append(
                    "VSH: Cannot calculate; GR range is insufficient"
                )

        else:

            status.append(
                "VSH: Not available; GR curve is required"
            )

    # --------------------------------------------------------
    # 2. PHIT: use existing curve or density porosity
    # --------------------------------------------------------

    phit = get_curve("PHIT")

    if phit is not None:

        phit = to_fraction(phit)
        status.append("PHIT: Existing LAS curve")

    else:

        rhob = get_curve("RHOB")

        if rhob is not None:

            # Convert kg/m3 to g/cm3 if necessary
            if rhob.dropna().median() > 100:

                rhob = rhob / 1000

            rho_matrix = params["rho_matrix"]
            rho_fluid = params["rho_fluid"]

            if rho_matrix > rho_fluid:

                phit = (
                    (rho_matrix - rhob) /
                    (rho_matrix - rho_fluid)
                ).clip(0, 1)

                df["PHIT_CALC"] = phit
                detected["PHIT"] = "PHIT_CALC"

                status.append(
                    "PHIT: Calculated using density porosity"
                )

        if phit is None:

            status.append(
                "PHIT: Not available; PHIT or RHOB is required"
            )

    # --------------------------------------------------------
    # 3. SW: use existing curve or calculate using Archie
    # --------------------------------------------------------

    sw = get_curve("SW")

    if sw is not None:

        sw = to_fraction(sw)
        status.append("SW: Existing LAS curve")

    else:

        rt = get_curve("RT")

        if (
            phit is not None
            and rt is not None
        ):

            rw = params["rw"]
            a = params["archie_a"]
            m = params["archie_m"]
            n = params["archie_n"]

            if rw > 0 and a > 0 and m > 0 and n > 0:

                valid = (
                    (phit > 0) &
                    (rt > 0)
                )

                sw = pd.Series(
                    np.nan,
                    index=df.index,
                    dtype=float
                )

                archie_value = (
                    a * rw /
                    (
                        phit[valid] ** m *
                        rt[valid]
                    )
                )

                sw.loc[valid] = (
                    archie_value ** (1 / n)
                )

                sw = sw.clip(0, 1)

                df["SW_CALC"] = sw
                detected["SW"] = "SW_CALC"

                status.append(
                    "SW: Calculated using Archie equation"
                )

            else:

                status.append(
                    "SW: Invalid Archie parameters"
                )

        else:

            status.append(
                "SW: Cannot calculate; PHIT and RT are required"
            )

    # --------------------------------------------------------
    # 4. SH: hydrocarbon saturation
    # --------------------------------------------------------

    if sw is not None:

        sh = 1 - sw

        df["SH_CALC"] = sh
        detected["SH"] = "SH_CALC"

        status.append(
            "SH: Calculated using SH = 1 - SW"
        )

    else:

        sh = get_curve("SH")

        if sh is not None:

            sh = to_fraction(sh)
            status.append("SH: Existing LAS curve")

        else:

            status.append(
                "SH: Cannot calculate without SW"
            )

    # --------------------------------------------------------
    # 5. PAY FLAG
    # --------------------------------------------------------

    if (
        vsh is not None
        and phit is not None
        and sw is not None
    ):

        vsh_cutoff = params["vsh_cutoff"]
        phit_cutoff = params["phit_cutoff"]
        sw_cutoff = params["sw_cutoff"]

        valid = (
            vsh.notna() &
            phit.notna() &
            sw.notna()
        )

        pay_flag = pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

        pay_condition = (
            (vsh <= vsh_cutoff) &
            (phit >= phit_cutoff) &
            (sw <= sw_cutoff)
        )

        pay_flag.loc[valid] = (
            pay_condition.loc[valid].astype(int)
        )

        df["PAY_FLAG"] = pay_flag
        detected["PAY"] = "PAY_FLAG"

        status.append(
            "PAY: Classified using VSH, PHIT and SW cut-offs"
        )

    else:

        status.append(
            "PAY: Cannot calculate; VSH, PHIT and SW are required"
        )

    # --------------------------------------------------------
    # 6. Identify continuous pay intervals
    # --------------------------------------------------------

    pay_intervals = []

    if "PAY_FLAG" in df.columns:

        depth_column = detected.get("DEPTH")

        if depth_column in df.columns:

            flags = df["PAY_FLAG"].fillna(0).to_numpy()
            depths = df[depth_column].to_numpy()

            start = None

            for i, flag in enumerate(flags):

                if flag == 1 and start is None:
                    start = i

                is_last = i == len(flags) - 1

                if start is not None and (
                    flag != 1 or is_last
                ):

                    end = i if flag == 1 else i - 1

                    pay_intervals.append({
                        "Top depth": depths[start],
                        "Base depth": depths[end],
                        "Thickness": abs(
                            depths[end] - depths[start]
                        )
                    })

                    start = None

    return df, detected, pay_intervals, status

# ============================================================
# 6. FILE PROCESSING
# ============================================================

IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".webp"
)

TEXT_EXTENSIONS = (
    ".json", ".txt", ".asc", ".dat"
)

WELL_LOG_EXTENSIONS = (
    ".las", ".csv", ".xlsx", ".xls"
)

SEISMIC_EXTENSIONS = (
    ".sgy", ".segy"
)


def process_uploaded_files(uploaded_files, selected_well=None):

    attachments = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name
        file_bytes = None
        extension = os.path.splitext(file_name)[1].lower()

        # ----------------------------------------
        # SEG-Y FILES
        # ----------------------------------------

        if extension in SEISMIC_EXTENSIONS:

            temp_path = None

            try:
                # Save uploaded SEG-Y temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=extension,
                    delete=False
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                # Read a manageable seismic preview
                seismic_data = read_segy_preview(
                    temp_path,
                    max_traces=1200,
                    sample_step=1
                )

                # Preserve the original filename
                seismic_data["filename"] = file_name

                # Store seismic data for interactive display
                st.session_state.seismic_section = seismic_data
                st.session_state.seismic_filename = file_name

                # Generate metadata summary for Gemma
                summary_df = get_seismic_summary({
                    **seismic_data,
                    "amplitude_stats": {
                        "minimum": float(
                            np.nanmin(seismic_data["amplitudes"])
                        ),
                        "maximum": float(
                            np.nanmax(seismic_data["amplitudes"])
                        ),
                        "mean": float(
                            np.nanmean(seismic_data["amplitudes"])
                        ),
                        "standard_deviation": float(
                            np.nanstd(seismic_data["amplitudes"])
                        ),
                        "rms": float(
                            np.sqrt(
                                np.nanmean(
                                    seismic_data["amplitudes"] ** 2
                                )
                            )
                        )
                    }
                })

                attachments.append({
                    "name": file_name,
                    "kind": "text",
                    "content": (
                        "SEG-Y seismic metadata:\n\n"
                        + summary_df.to_string(index=False)
                        + "\n\nThe interactive seismic section "
                        "has been loaded in the Geological Workspace."
                    )
                })

                st.success(
                    f"SEG-Y seismic preview loaded: {file_name}"
                )

            except Exception as e:

                st.error(
                    f"Unable to process SEG-Y file {file_name}: {e}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            continue

        # ----------------------------------------
        # LAS, CSV AND EXCEL WELL-LOG FILES
        # ----------------------------------------

        if extension in WELL_LOG_EXTENSIONS:
            file_bytes = uploaded_file.getvalue()
            try:

                if extension == ".las":

                    las_text = file_bytes.decode(
                        "utf-8",
                        errors="replace"
                    )

                    las = lasio.read(
                        io.StringIO(las_text)
                    )

                    df_log = las.df().reset_index()

                elif extension == ".csv":

                    # Treat common LAS null values as missing data
                    df_log = pd.read_csv(
                        io.BytesIO(file_bytes),
                        na_values=[-999.25, -999]
                    )

                else:

                    df_log = pd.read_excel(
                        io.BytesIO(file_bytes),
                        na_values=[-999.25, -999]
                    )

                # Standardise column names
                df_log.columns = (
                    df_log.columns
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                # Recognise common depth names
                if "DEPTH" not in df_log.columns:

                    for alias in ["DEPT", "MD"]:

                        if alias in df_log.columns:

                            df_log.rename(
                                columns={alias: "DEPTH"},
                                inplace=True
                            )

                            break

                # Detect available curves
                detected_curves = detect_well_log_curves(
                    df_log
                )

                if "DEPTH" not in detected_curves:

                    st.warning(
                        f"{file_name}: No recognised depth column. "
                        "The file cannot be plotted as a depth-based log."
                    )

                    continue

                # Convert detected curves to numeric values
                for column in detected_curves.values():

                    df_log[column] = pd.to_numeric(
                        df_log[column],
                        errors="coerce"
                    )

                # ----------------------------------------
                # AUTOMATIC PETROPHYSICAL CALCULATIONS
                # ----------------------------------------

                df_log, detected_curves, pay_intervals, petro_status = (
                    calculate_petrophysical_curves(
                        df_log,
                        detected_curves,
                        PETRO_PARAMS
                    )
                )

                # Save processed well-log data
                st.session_state.well_log_df = df_log
                st.session_state.detected_curves = detected_curves
                st.session_state.pay_intervals = pay_intervals
                st.session_state.petro_status = petro_status

                st.success(
                    f"Well-log data and petrophysical calculations "
                    f"completed: {file_name}"
                )

                st.write(
                    "Detected and calculated curves:",
                    list(detected_curves.keys())
                )

                for item in petro_status:
                    st.caption("• " + item)

                attachments.append({
                    "name": file_name,
                    "kind": "text",
                    "content": (
                        "Automatically detected and calculated "
                        "well-log curves:\n"
                        + str(detected_curves)
                        + "\n\nPetrophysical calculation status:\n"
                        + "\n".join(petro_status)
                        + "\n\n"
                        + df_log.head(500).to_csv(index=False)
                    )
                })

            except Exception as e:

                st.error(
                    f"Unable to process {file_name}: {e}"
                )

            continue

        # ----------------------------------------
        # IMAGE FILES
        # ----------------------------------------

        if extension in IMAGE_EXTENSIONS:
            file_bytes = uploaded_file.getvalue()
            mime_type = (
                mimetypes.guess_type(file_name)[0]
                or "image/jpeg"
            )

            attachments.append({
                "name": file_name,
                "kind": "image",
                "image_bytes": file_bytes,
                "mime_type": mime_type
            })

            continue

        # ----------------------------------------
        # TEXT FILES
        # ----------------------------------------

        if extension in TEXT_EXTENSIONS:
            file_bytes = uploaded_file.getvalue()
            extracted_text = file_bytes.decode(
                "utf-8",
                errors="replace"
            )

            attachments.append({
                "name": file_name,
                "kind": "text",
                "content": extracted_text[:30000]
            })

            continue

        # ----------------------------------------
        # UNSUPPORTED FILES
        # ----------------------------------------

        attachments.append({
            "name": file_name,
            "kind": "unsupported",
            "content": "This file format is not currently supported."
        })

    return attachments


# ============================================================
# 6A. INTERACTIVE SEISMIC IMAGE VIEWER
# ============================================================

def render_zoomable_image(image_bytes, image_name):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    mime_type = mimetypes.guess_type(image_name)[0] or "image/png"

    safe_name = escape(image_name)

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #F4F3E9;
        }

        .viewer {
            width: 100%;
            background: #E8E9DC;
            border: 1px solid #C9CEB8;
            border-radius: 12px;
            overflow: hidden;
        }

        .toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
            background: #354D35;
        }

        .toolbar button {
            background: #E9EBD9;
            color: #354D35;
            border: none;
            border-radius: 6px;
            padding: 8px 13px;
            font-weight: bold;
            cursor: pointer;
        }

        .toolbar button:hover {
            background: #C9D3AE;
        }

        #zoomLabel {
            color: white;
            font-size: 13px;
            margin-left: auto;
        }

        .canvas-container {
            width: 100%;
            height: 610px;
            overflow: hidden;
            background: #F8F8F2;
            cursor: grab;
        }

        .canvas-container:active {
            cursor: grabbing;
        }

        canvas {
            width: 100%;
            height: 100%;
            display: block;
            touch-action: none;
        }

        .image-name {
            padding: 8px 12px;
            color: #354D35;
            font-size: 13px;
            font-weight: bold;
            background: #E8E9DC;
        }
    </style>
    </head>

    <body>

    <div class="viewer">

        <div class="toolbar">

            <button onclick="zoomIn()">＋ Zoom In</button>

            <button onclick="zoomOut()">－ Zoom Out</button>

            <button onclick="resetZoom()">⟲ Reset</button>

            <span id="zoomLabel">100%</span>

        </div>

        <div class="canvas-container" id="container">
            <canvas id="seismicCanvas"></canvas>
        </div>

        <div class="image-name">IMAGE_NAME</div>

    </div>

    <script>

        const canvas = document.getElementById("seismicCanvas");
        const container = document.getElementById("container");
        const ctx = canvas.getContext("2d");

        const img = new Image();

        img.src = "data:IMAGE_MIME;base64,IMAGE_DATA";

        let zoom = 1;
        let panX = 0;
        let panY = 0;

        let dragging = false;
        let lastX = 0;
        let lastY = 0;

        let fitScale = 1;

        function resizeCanvas() {

            const rect = container.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;

            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            if (img.complete && img.naturalWidth > 0) {
                drawImage();
            }
        }

        function drawImage() {

            const width = container.clientWidth;
            const height = container.clientHeight;

            if (!img.naturalWidth) return;

            fitScale = Math.min(
                (width - 30) / img.width,
                (height - 30) / img.height
            );

            const scale = fitScale * zoom;

            const drawWidth = img.width * scale;
            const drawHeight = img.height * scale;

            const x = (width - drawWidth) / 2 + panX;
            const y = (height - drawHeight) / 2 + panY;

            ctx.clearRect(0, 0, width, height);

            ctx.drawImage(
                img,
                x,
                y,
                drawWidth,
                drawHeight
            );

            document.getElementById("zoomLabel").textContent =
                Math.round(zoom * 100) + "%";
        }

        function changeZoom(newZoom, mouseX, mouseY) {

            newZoom = Math.max(0.25, Math.min(newZoom, 5));

            const width = container.clientWidth;
            const height = container.clientHeight;

            const oldScale = fitScale * zoom;
            const newScale = fitScale * newZoom;

            if (mouseX !== undefined && mouseY !== undefined) {

                const oldX = (width - img.width * oldScale) / 2 + panX;
                const oldY = (height - img.height * oldScale) / 2 + panY;

                const imageX = (mouseX - oldX) / oldScale;
                const imageY = (mouseY - oldY) / oldScale;

                panX = mouseX - imageX * newScale -
                    (width - img.width * newScale) / 2;

                panY = mouseY - imageY * newScale -
                    (height - img.height * newScale) / 2;
            }

            zoom = newZoom;
            drawImage();
        }

        function zoomIn() {
            changeZoom(zoom * 1.2);
        }

        function zoomOut() {
            changeZoom(zoom / 1.2);
        }

        function resetZoom() {
            zoom = 1;
            panX = 0;
            panY = 0;
            drawImage();
        }

        // Mouse-wheel zoom
        container.addEventListener("wheel", function(event) {

            event.preventDefault();

            const rect = container.getBoundingClientRect();

            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;

            if (event.deltaY < 0) {
                changeZoom(zoom * 1.12, mouseX, mouseY);
            } else {
                changeZoom(zoom / 1.12, mouseX, mouseY);
            }

        }, { passive: false });

        // Drag to move seismic image
        canvas.addEventListener("pointerdown", function(event) {

            dragging = true;

            lastX = event.clientX;
            lastY = event.clientY;

            canvas.setPointerCapture(event.pointerId);

        });

        canvas.addEventListener("pointermove", function(event) {

            if (!dragging) return;

            panX += event.clientX - lastX;
            panY += event.clientY - lastY;

            lastX = event.clientX;
            lastY = event.clientY;

            drawImage();

        });

        canvas.addEventListener("pointerup", function() {
            dragging = false;
        });

        canvas.addEventListener("pointercancel", function() {
            dragging = false;
        });

        img.onload = function() {
            resizeCanvas();
        };

        new ResizeObserver(resizeCanvas).observe(container);

    </script>

    </body>
    </html>
    """

    # Replace HTML placeholders
    html_code = html_code.replace(
        "IMAGE_DATA", encoded_image
    ).replace(
        "IMAGE_NAME", safe_name
    ).replace(
        "IMAGE_MIME", mime_type
    )

    # Display interactive viewer
    components.html(
        html_code,
        height=665,
        scrolling=False
    )


# ============================================================
# CONVERT SEISMIC AMPLITUDES INTO AN IMAGE FOR GEMMA
# ============================================================

def create_seismic_ai_image(seismic_data):

    amplitudes = np.asarray(
        seismic_data["amplitudes"],
        dtype=np.float32
    )

    # Replace invalid values
    amplitudes = np.nan_to_num(
        amplitudes,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Use percentile clipping to improve seismic visibility
    clip_value = np.percentile(
        np.abs(amplitudes),
        99.5
    )

    if clip_value <= 0:
        clip_value = 1.0

    amplitudes = np.clip(
        amplitudes,
        -clip_value,
        clip_value
    )

    # Normalise signed seismic amplitudes to 8-bit grayscale
    normalised = (
        (amplitudes / clip_value + 1.0) * 127.5
    )

    image_array = normalised.astype(np.uint8)

    # Transpose so traces are horizontal-axis positions
    image_array = image_array.T

    # Create image (uint8 2D arrays are treated as grayscale)
    seismic_image = Image.fromarray(image_array)

    # Resize to a manageable image size
    target_width = min(
        1400,
        max(800, seismic_image.width)
    )

    target_height = int(
        seismic_image.height *
        target_width /
        seismic_image.width
    )

    seismic_image = seismic_image.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS
    )

    image_buffer = io.BytesIO()

    seismic_image.save(
        image_buffer,
        format="PNG"
    )

    return image_buffer.getvalue()


def extract_selected_seismic_region(
    seismic_data,
    x_range,
    y_range
):
    """
    Extract a selected trace/time region from the original
    seismic amplitude matrix.
    """

    amplitudes = np.asarray(
        seismic_data["amplitudes"],
        dtype=np.float32
    )

    trace_axis = np.asarray(
        seismic_data["trace_axis"],
        dtype=float
    )

    sample_axis = np.asarray(
        seismic_data["sample_axis"],
        dtype=float
    )

    x_min, x_max = sorted(x_range)
    y_min, y_max = sorted(y_range)

    trace_indices = np.where(
        (trace_axis >= x_min) &
        (trace_axis <= x_max)
    )[0]

    sample_indices = np.where(
        (sample_axis >= y_min) &
        (sample_axis <= y_max)
    )[0]

    if len(trace_indices) < 2 or len(sample_indices) < 2:
        raise ValueError(
            "Selected region is too small. Select a larger area."
        )

    cropped_amplitudes = amplitudes[
        np.ix_(trace_indices, sample_indices)
    ]

    cropped_data = {
        "amplitudes": cropped_amplitudes,
        "trace_axis": trace_axis[trace_indices],
        "sample_axis": sample_axis[sample_indices],
        "trace_count": len(trace_indices),
        "sample_count": len(sample_indices),
        "filename": seismic_data["filename"]
    }

    return cropped_data


def create_selected_seismic_png(seismic_data):
    """
    Convert the selected amplitude region into a PNG.
    """

    amplitudes = np.asarray(
        seismic_data["amplitudes"],
        dtype=np.float32
    )

    amplitudes = np.nan_to_num(
        amplitudes,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    clip_value = np.percentile(
        np.abs(amplitudes),
        99.5
    )

    if clip_value <= 0:
        clip_value = 1.0

    normalised = (
        (np.clip(
            amplitudes,
            -clip_value,
            clip_value
        ) / clip_value + 1.0) * 127.5
    )

    image_array = normalised.T.astype(np.uint8)

    image = Image.fromarray(image_array)

    target_width = min(1600, max(800, image.width))

    target_height = int(
        image.height * target_width / image.width
    )

    image = image.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()

# -----------------------------------------------------------------------------
# DATA SETUP (1D WELL LOG)
# -----------------------------------------------------------------------------
ALL_WELLS = [f"Well {w}" for w in ["P-1", "M-1", "A-2", "A-3", "B-1", "B-2", "C-1", "C-2", "D-1", "D-2", "E-1", "E-2", "F-1"]]

TRAP_METADATA = {
    "Well P-1": {
        "system": "Structural Trap",
        "type": "Normal Fault Block",
        "badge_style": "trap-badge-struct",
        "has_seismic": True,
        "seismic_img": "assets/p1_seismic.png",
    },
    "Well M-1": {
        "system": "Stratigraphic Trap",
        "type": "Angular Unconformity / Channel Sand",
        "badge_style": "trap-badge-strati",
        "has_seismic": True,
        "seismic_img": "assets/m1_seismic.png",
    }
}

for well in ALL_WELLS:
    if well not in TRAP_METADATA:
        TRAP_METADATA[well] = {
            "system": "Evaluation Pending",
            "type": "Unclassified Outpost",
            "badge_style": "trap-badge-unknown",
            "has_seismic": False,
            "seismic_img": None
        }


def render_4track_plotly_light(df_log=None):

    if df_log is None:
        return None

    detected = st.session_state.get(
        "detected_curves", {}
    )

    depth_column = detected.get("DEPTH", "DEPTH")

    if depth_column not in df_log.columns:
        return None

    # Display up to five curves in two columns
    preferred_curves = [
        "GR", "RT", "RHOB", "NPHI", "DT",
        "VSH", "PHIT", "SW", "SH", "PAY"
    ]

    available_curves = [
        curve for curve in preferred_curves
        if curve in detected
        and detected[curve] in df_log.columns
    ][:5]

    if not available_curves:
        return None

    colours = {
        "GR": "#2e7d32",
        "RT": "#1565c0",
        "RHOB": "#c62828",
        "NPHI": "#00838f",
        "DT": "#6a1b9a",
        "VSH": "#558b2f",
        "PHIT": "#ef6c00",
        "SW": "#ad1457",
        "SH": "#4527a0",
        "PAY": "#d97706"
    }

    number_of_rows = (len(available_curves) + 1) // 2

    subplot_specs = []

    for row in range(number_of_rows):

        if row == number_of_rows - 1 and len(available_curves) % 2 == 1:
            subplot_specs.append([{}, None])
        else:
            subplot_specs.append([{}, {}])

    fig = make_subplots(
        rows=number_of_rows,
        cols=2,
        shared_yaxes=True,
        subplot_titles=available_curves,
        vertical_spacing=0.15,
        horizontal_spacing=0.13,
        specs=subplot_specs
    )

    for i, curve in enumerate(available_curves):

        row = (i // 2) + 1
        col = (i % 2) + 1

        column_name = detected[curve]

        fig.add_trace(
            go.Scatter(
                x=df_log[column_name],
                y=df_log[depth_column],
                mode="lines",
                name=curve,
                line=dict(
                    color=colours.get(curve, "#333333"),
                    width=1.3,
                    shape="hv" if curve == "PAY" else "linear"
                ),
                connectgaps=False
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(
            title_text=None,
            showgrid=True,
            gridcolor="#D0D0D0",
            showline=True,
            mirror=True,
            zeroline=False,
            row=row,
            col=col
        )

        fig.update_yaxes(
            autorange="reversed",
            showgrid=True,
            gridcolor="#C8C8C8",
            showline=True,
            mirror=True,
            zeroline=False,
            row=row,
            col=col
        )

    fig.update_layout(
        height=max(850, number_of_rows * 430),
        margin=dict(l=55, r=20, t=90, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(color="black", size=10),
        hovermode="y"
    )

    fig.update_yaxes(
        title_text="Depth (m)",
        row=1,
        col=1
    )

    return fig


# ============================================================
# 7. BUILD GEMMA API MESSAGES
# ============================================================

def build_api_messages(messages):

    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Only the most recent user message keeps its images.
    # Older images are replaced by a short note so the request
    # does not grow with every turn.
    last_user_index = max(
        (
            index for index, item in enumerate(messages)
            if item["role"] == "user"
        ),
        default=-1
    )

    for message_index, message in enumerate(messages):

        if message["role"] == "assistant":

            api_messages.append({
                "role": "assistant",
                "content": message.get("text", "")
            })

            continue

        # ----------------------------------------
        # USER MESSAGE
        # ----------------------------------------

        content = []

        user_text = message.get("text", "")

        content.append({
            "type": "text",
            "text": user_text
        })

        # ----------------------------------------
        # ATTACHMENTS
        # ----------------------------------------

        for attachment in message.get("attachments", []):

            if attachment["kind"] == "image":

                if message_index != last_user_index:

                    content.append({
                        "type": "text",
                        "text": (
                            f"\n\n[Earlier image omitted: "
                            f"{attachment['name']}]"
                        )
                    })

                    continue

                image_bytes = attachment["image_bytes"]

                encoded_image = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                mime_type = attachment.get(
                    "mime_type",
                    "image/jpeg"
                )

                image_url = (
                    f"data:{mime_type};base64,{encoded_image}"
                )

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })

            else:

                content.append({
                    "type": "text",
                    "text": (
                        f"\n\nFILE: {attachment['name']}\n"
                        f"{attachment['content']}"
                    )
                })

        api_messages.append({
            "role": "user",
            "content": content
        })

    return api_messages


# ============================================================
# 8. GENERATE GEMMA RESPONSE
# ============================================================

def generate_ai_response(messages, selected_well):

    api_messages = build_api_messages(messages)

    # Retrieve relevant geological knowledge
    latest_question = next(
        (
            message.get("text", "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        ""
    )

    geological_context = format_resource_context(
        latest_question,
        max_results=4
    )

    # Add retrieved knowledge to Gemma's system instructions
    api_messages[0]["content"] += (
        "\n\n"
        "GEOLOGICAL REFERENCE LIBRARY:\n"
        "Use the following references to support your interpretation. "
        "Compare the visible seismic evidence with the descriptions. "
        "Do not force a match if the evidence is insufficient. "
        "Distinguish observations from interpretations.\n\n"
        + geological_context
    )

    if selected_well and selected_well != "All wells":
        api_messages[0]["content"] += (
            f"\n\nCurrently selected well: {selected_well}."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMMA_API_KEY}"
    }

    payload = {
        "model": GEMMA_MODEL,
        "messages": api_messages,
        "temperature": 0.2,
        "max_tokens": 2000
    }

    response = requests.post(
        GEMMA_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    answer = result["choices"][0]["message"]["content"]

    if not answer:
        answer = "Gemma returned an empty response."

    return answer


# ============================================================
# 9. LEFT SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="color:#52663C;">
            🌍 StrataMind
        </h2>
        <p style="color:#888B80;font-size:13px;">
            Subsurface Exploration Copilot
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # New conversation button
    if st.button(
        "＋ New Conversation",
        use_container_width=True
    ):

        create_new_conversation()
        st.rerun()

    st.markdown("### Conversation History")

    for conversation_id, conversation_data in reversed(
        list(st.session_state.conversations.items())
    ):

        is_active = (
            conversation_id ==
            st.session_state.active_conversation
        )

        label = conversation_data["title"]

        if len(label) > 25:
            label = label[:25] + "..."

        if st.button(
            f"{'● ' if is_active else '○ '}{label}",
            key=f"conversation_{conversation_id}",
            use_container_width=True
        ):

            st.session_state.active_conversation = conversation_id
            st.rerun()

    st.markdown("---")

    st.markdown("### Geological Data")

    selected_well = st.text_input(
        "Selected well / dataset",
        value="All wells",
        help="Well selection will be connected to well_knowledge.py later."
    )

    st.markdown("---")

    with st.expander("Petrophysical Parameters"):

        st.caption(
            "Adjust these values to match your formation. "
            "The initial values are illustrative only."
        )

        rw = st.number_input(
            "Formation water resistivity (Rw), ohm.m",
            min_value=0.0001,
            value=0.10,
            step=0.01
        )

        archie_a = st.number_input(
            "Archie a",
            min_value=0.01,
            value=1.0,
            step=0.1
        )

        archie_m = st.number_input(
            "Cementation exponent (m)",
            min_value=0.1,
            value=2.0,
            step=0.1
        )

        archie_n = st.number_input(
            "Saturation exponent (n)",
            min_value=0.1,
            value=2.0,
            step=0.1
        )

        rho_matrix = st.number_input(
            "Matrix density (g/cm3)",
            min_value=0.1,
            value=2.65,
            step=0.05
        )

        rho_fluid = st.number_input(
            "Fluid density (g/cm3)",
            min_value=0.1,
            value=1.00,
            step=0.05
        )

        st.markdown("**Pay cut-offs**")

        vsh_cutoff = st.number_input(
            "Maximum VSH",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05
        )

        phit_cutoff = st.number_input(
            "Minimum PHIT",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

        sw_cutoff = st.number_input(
            "Maximum SW",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05
        )

    PETRO_PARAMS = {
        "rw": rw,
        "archie_a": archie_a,
        "archie_m": archie_m,
        "archie_n": archie_n,
        "rho_matrix": rho_matrix,
        "rho_fluid": rho_fluid,
        "vsh_cutoff": vsh_cutoff,
        "phit_cutoff": phit_cutoff,
        "sw_cutoff": sw_cutoff
    }

    if st.button(
        "🗑️ Clear Current Conversation",
        use_container_width=True
    ):

        current_conversation()["messages"] = []
        current_conversation()["title"] = "New conversation"

        st.rerun()


# ============================================================
# 10. MAIN PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-title">
        Subsurface Exploration <span>Copilot</span>
    </div>

    <div class="hero-subtitle">
        Your AI Geoscientist for Seismic Interpretation,
        Trap Identification and Exploration Reasoning
    </div>
    """,
    unsafe_allow_html=True
)

conversation = current_conversation()

# ============================================================
# 11. THREE-PANEL MAIN WORKSPACE
# ============================================================

well_col, seismic_col, chat_col = st.columns(
    [1.0, 1.5, 1.2],
    gap="medium"
)

question = None
uploaded_files = []

# ============================================================
# 12. LEFT PANEL: 1D WELL LOG EXPLORER
# ============================================================

with well_col:

    with st.container(border=True):

        st.markdown("### 📊 1D Well Log Explorer")

        st.caption(
            "Upload and analyse your well-log data."
        )

        # ----------------------------------------------------
        # WELL LOG UPLOAD
        # ----------------------------------------------------

        st.markdown("#### Upload Your LAS File Here")

        well_file = st.file_uploader(
            "Choose a well-log file",
            type=["las", "csv", "xlsx", "xls"],
            accept_multiple_files=False,
            key="well_log_upload",
            help="Upload a LAS, CSV or Excel well-log file."
        )

        if well_file is not None:

            st.caption(
                f"Selected file: {well_file.name}"
            )

            if st.button(
                "Load Well Log",
                use_container_width=True,
                key="load_well_log"
            ):

                with st.spinner(
                    "Reading well-log data and calculating petrophysical properties..."
                ):

                    process_uploaded_files(
                        [well_file],
                        selected_well=selected_well
                    )

                if st.session_state.well_log_df is not None:

                    st.session_state.well_log_filename = (
                        well_file.name
                    )

                    st.success(
                        "Well-log data loaded successfully."
                    )

                    st.rerun()

        # ----------------------------------------------------
        # DISPLAY LOADED WELL
        # ----------------------------------------------------

        well_log_df = st.session_state.well_log_df

        if well_log_df is not None:

            st.divider()

            st.markdown("#### Loaded Well")

            st.success(
                st.session_state.get(
                    "well_log_filename",
                    "Well-log dataset loaded"
                )
            )

            depth_column = st.session_state.detected_curves.get(
                "DEPTH", "DEPTH"
            )

            if depth_column in well_log_df.columns:

                depth_min = well_log_df[depth_column].min()
                depth_max = well_log_df[depth_column].max()

                st.caption(
                    f"Depth range: {depth_min:.2f} – {depth_max:.2f}"
                )

            # ------------------------------------------------
            # WELL LOG CURVES
            # ------------------------------------------------

            st.markdown("#### Well Log Curves")

            well_fig = render_4track_plotly_light(
                well_log_df
            )

            if well_fig is not None:

                st.plotly_chart(
                    well_fig,
                    use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "responsive": True,
                        "displaylogo": False
                    },
                    key="well_log_plot"
                )

                with st.expander("Well-Log Curve Glossary"):

                    curve_definitions = {
                        "GR": "Gamma Ray: indicates natural radioactivity, commonly used to estimate shale content.",
                        "RT": "Resistivity: measures resistance to electrical current, useful for evaluating formation fluids.",
                        "RHOB": "Bulk Density: measures formation bulk density.",
                        "NPHI": "Neutron Porosity: estimates formation porosity based on hydrogen concentration.",
                        "DT": "Sonic Travel Time: measures the time taken for an acoustic wave to travel through the formation.",
                        "VSH": "Volume of Shale: estimated proportion of shale in the formation.",
                        "PHIT": "Total Porosity: estimated total pore-space fraction.",
                        "SW": "Water Saturation: estimated fraction of pore space occupied by water.",
                        "SH": "Hydrocarbon Saturation: estimated fraction of pore space occupied by hydrocarbons.",
                        "PAY": "Pay Indicator: identifies intervals satisfying the selected petrophysical cut-offs."
                    }

                    glossary_shown = False

                    for curve in st.session_state.detected_curves:

                        if curve in curve_definitions:

                            st.markdown(
                                f"**{curve}**: {curve_definitions[curve]}"
                            )

                            glossary_shown = True

                    if not glossary_shown:

                        st.warning(
                            "No recognised well-log curves were found."
                        )

            # ------------------------------------------------
            # PETROPHYSICAL ANALYSIS
            # ------------------------------------------------

            st.divider()

            st.markdown("#### Petrophysical Analysis")

            for item in st.session_state.get(
                "petro_status", []
            ):
                st.caption("• " + item)

            pay_intervals = st.session_state.get(
                "pay_intervals", []
            )

            if pay_intervals:

                st.markdown("**Identified Pay Intervals**")

                pay_df = pd.DataFrame(pay_intervals)

                st.dataframe(
                    pay_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                total_pay = pay_df["Thickness"].sum()

                st.metric(
                    "Total Identified Pay Thickness",
                    f"{total_pay:.2f} m"
                )

            else:

                st.info(
                    "No pay intervals identified, or insufficient "
                    "well-log curves to perform pay classification."
                )

        else:

            st.info(
                "Upload your LAS file to start exploring "
                "the well-log curves."
            )

# ============================================================
# 13. CENTRE PANEL: 2D SEISMIC EXPLORER
# ============================================================

with seismic_col:

    with st.container(border=True):

        st.markdown("### 🌊 2D Seismic Explorer")

        st.caption(
            "Explore seismic sections from the HPC datasets "
            "or load your own SEG-Y file."
        )

        # ----------------------------------------------------
        # DATA SOURCE SELECTION
        # ----------------------------------------------------

        seismic_source = st.radio(
            "Select data source",
            [
                "HPC Datasets",
                "Upload Own SEG-Y"
            ],
            horizontal=True,
            key="seismic_source"
        )

        # ----------------------------------------------------
        # OPTION A: EXISTING HPC DATASETS
        # ----------------------------------------------------

        if seismic_source == "HPC Datasets":

            SEGY_FOLDER = (
                "/data/epic20/APGCE/input_data/"
                "challenge1_challenge2/stack/"
            )

            SEGY_FILES = {
                "Far Stack":
                    "Challenge1_Challenge2_Far_Stack_cropped.sgy",

                "Full Stack":
                    "Challenge1_Challenge2_Full_Stack_cropped.sgy",

                "Mid Stack":
                    "Challenge1_Challenge2_Mid_Stack_cropped.sgy",

                "Near Stack":
                    "Challenge1_Challenge2_Near_Stack_cropped.sgy",

                "Ultra Far Stack":
                    "Challenge1_Challenge2_UltraFar_Stack_cropped.sgy"
            }

            selected_stack = st.selectbox(
                "Choose HPC seismic dataset",
                list(SEGY_FILES.keys()),
                key="hpc_seismic_selection"
            )

            if st.button(
                "Load Seismic Dataset",
                use_container_width=True,
                key="load_hpc_seismic"
            ):

                selected_path = os.path.join(
                    SEGY_FOLDER,
                    SEGY_FILES[selected_stack]
                )

                if os.path.exists(selected_path):

                    try:

                        with st.spinner(
                            "Reading seismic preview from HPC..."
                        ):

                            seismic_data = read_segy_preview(
                                selected_path,
                                max_traces=1200,
                                sample_step=1
                            )

                            st.session_state.seismic_section = (
                                seismic_data
                            )

                            st.session_state.seismic_filename = (
                                seismic_data["filename"]
                            )

                        st.success(
                            "Seismic dataset loaded successfully."
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to load SEG-Y dataset: {e}"
                        )

                else:

                    st.error(
                        "The selected HPC dataset could not be found."
                    )

        # ----------------------------------------------------
        # OPTION B: CLIENT'S OWN SEG-Y FILE
        # ----------------------------------------------------

        else:

            st.info(
                "Upload a SEG-Y file from your computer."
            )

            st.caption(
                "Note: This initial uploader is suitable for "
                "moderate-sized files. The 5 GB HPC datasets "
                "should be loaded using the HPC option. "
                "A separate chunked uploader will be needed "
                "for very large client files."
            )

            own_seismic_file = st.file_uploader(
                "Choose your SEG-Y file",
                type=["sgy", "segy"],
                accept_multiple_files=False,
                key="own_seismic_upload"
            )

            if st.button(
                "Load Uploaded SEG-Y",
                use_container_width=True,
                key="load_own_seismic"
            ):

                if own_seismic_file is None:

                    st.warning(
                        "Please select a SEG-Y file first."
                    )

                else:

                    temp_path = None

                    try:

                        with st.spinner(
                            "Reading uploaded seismic data..."
                        ):

                            suffix = (
                                os.path.splitext(
                                    own_seismic_file.name
                                )[1].lower()
                                or ".sgy"
                            )

                            with tempfile.NamedTemporaryFile(
                                suffix=suffix,
                                delete=False
                            ) as temp_file:

                                temp_file.write(
                                    own_seismic_file.getbuffer()
                                )

                                temp_path = temp_file.name

                            seismic_data = read_segy_preview(
                                temp_path,
                                max_traces=1200,
                                sample_step=1
                            )

                            seismic_data["filename"] = (
                                own_seismic_file.name
                            )

                            seismic_data["filepath"] = None

                            st.session_state.seismic_section = (
                                seismic_data
                            )

                            st.session_state.seismic_filename = (
                                own_seismic_file.name
                            )

                        st.success(
                            "Your seismic file has been loaded."
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to read uploaded SEG-Y: {e}"
                        )

                    finally:

                        if (
                            temp_path
                            and os.path.exists(temp_path)
                        ):
                            os.remove(temp_path)

        # ----------------------------------------------------
        # DISPLAY LOADED SEISMIC DATA
        # ----------------------------------------------------

        seismic_data = st.session_state.get(
            "seismic_section"
        )

        if seismic_data is not None:

            st.divider()

            st.success(
                f"Loaded: {seismic_data['filename']}"
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Total Traces",
                f"{seismic_data['trace_count']:,}"
            )

            metric2.metric(
                "Samples per Trace",
                f"{seismic_data['sample_count']:,}"
            )

            st.caption(
                f"Displaying "
                f"{seismic_data['displayed_traces']:,} "
                "selected traces."
            )

            # DISPLAY SETTINGS

            with st.expander("Seismic Display Settings"):

                clip_percentile = st.slider(
                    "Amplitude Contrast",
                    min_value=90.0,
                    max_value=100.0,
                    value=99.5,
                    step=0.5,
                    key="seismic_clip"
                )

                colour_scale = st.selectbox(
                    "Seismic Colour Scale",
                    [
                        "RdBu",
                        "Greys",
                        "RdYlBu",
                        "Jet",
                        "Viridis"
                    ],
                    index=0,
                    key="seismic_colour"
                )

            # INTERACTIVE SEISMIC DISPLAY
            seismic_fig = plot_seismic_section(
                seismic_data,
                clip_percentile=clip_percentile,
                colorscale=colour_scale
            )

            # ----------------------------------------------------
            # ADD WELL-TO-SEISMIC CORRELATION ANNOTATIONS
            # ----------------------------------------------------

            correlation_data = st.session_state.get(
                "well_seismic_correlation"
            )

            if correlation_data is not None:

                # Check that the overlay belongs to the current datasets
                same_well = (
                    correlation_data["well_filename"]
                    == st.session_state.get("well_log_filename")
                )

                same_seismic = (
                    correlation_data["seismic_filename"]
                    == seismic_data["filename"]
                )

                if same_well and same_seismic:

                    seismic_fig = add_well_correlation_overlay(
                        seismic_fig,
                        seismic_data,
                        correlation_data
                    )

                else:

                    st.warning(
                        "The correlation overlay belongs to a different "
                        "well or seismic dataset. Generate a new correlation."
                    )

            # ============================================================
            # SEISMIC AREA SELECTION FOR GEMMA
            # ============================================================

            st.markdown("### AI Geological Area Selection")

            st.caption(
                "Activate selection mode and drag a rectangle "
                "over the seismic section to analyse a specific area."
            )

            selection_mode = st.toggle(
                "Enable Area Selection",
                value=False,
                key="enable_seismic_area_selection"
            )

            selection_fig = go.Figure(seismic_fig)

            selection_fig.update_layout(
                dragmode="select" if selection_mode else "zoom"
            )

            selected_points = []

            if selection_mode:

                trace_axis = np.asarray(
                    seismic_data["trace_axis"]
                )

                sample_axis = np.asarray(
                    seismic_data["sample_axis"]
                )

                trace_step = max(
                    1,
                    len(trace_axis) // 180
                )

                sample_step = max(
                    1,
                    len(sample_axis) // 120
                )

                selection_traces = trace_axis[::trace_step]
                selection_samples = sample_axis[::sample_step]

                xx, yy = np.meshgrid(
                    selection_traces,
                    selection_samples
                )

                selection_fig.add_trace(
                    go.Scattergl(
                        x=xx.ravel(),
                        y=yy.ravel(),
                        mode="markers",
                        marker=dict(
                            size=7,
                            color="rgba(0,0,0,0.01)"
                        ),
                        name="Selection layer",
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            plot_event = st.plotly_chart(
                selection_fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False,
                    "doubleClick": "reset",
                    "modeBarButtonsToAdd": [
                        "zoomIn2d",
                        "zoomOut2d",
                        "resetScale2d",
                        "pan2d"
                    ]
                },
                key="interactive_seismic_plot",
                on_select="rerun",
                selection_mode=("box",)
            )

            if plot_event and plot_event.selection:
                selected_points = plot_event.selection.points

            # ============================================================
            # STORE SELECTED SEISMIC AREA
            # ============================================================

            if selected_points and selection_mode:

                selected_x = [
                    point["x"]
                    for point in selected_points
                    if point.get("x") is not None
                ]

                selected_y = [
                    point["y"]
                    for point in selected_points
                    if point.get("y") is not None
                ]

                if selected_x and selected_y:

                    x_range = [
                        min(selected_x),
                        max(selected_x)
                    ]

                    y_range = [
                        min(selected_y),
                        max(selected_y)
                    ]

                    st.session_state.selected_seismic_x_range = x_range
                    st.session_state.selected_seismic_y_range = y_range

                    st.success(
                        "Seismic area selected successfully."
                    )

            # ============================================================
            # DISPLAY SELECTED AREA
            # ============================================================

            if (
                st.session_state.get("selected_seismic_x_range") is not None
                and st.session_state.get("selected_seismic_y_range") is not None
            ):

                st.write(
                    "Selected trace range:",
                    st.session_state.selected_seismic_x_range
                )

                st.write(
                    "Selected time range (ms):",
                    st.session_state.selected_seismic_y_range
                )

                if st.button(
                    "Reset Selected Area",
                    key="reset_seismic_selection"
                ):

                    st.session_state.selected_seismic_x_range = None
                    st.session_state.selected_seismic_y_range = None

                    st.rerun()

                # ========================================================
                # RUN GEOLOGICAL INTERPRETATION
                # ========================================================

                if st.button(
                    "Run Geological Interpretation",
                    type="primary",
                    use_container_width=True,
                    key="analyse_selected_seismic_area"
                ):

                    try:

                        with st.spinner(
                            "Gemma is analysing the selected seismic region..."
                        ):

                            # 1. EXTRACT SELECTED SEISMIC AREA

                            cropped_data = extract_selected_seismic_region(
                                seismic_data,
                                st.session_state.selected_seismic_x_range,
                                st.session_state.selected_seismic_y_range
                            )

                            # 2. CONVERT SELECTED AREA INTO PNG

                            cropped_png = create_selected_seismic_png(
                                cropped_data
                            )

                            if not cropped_png:
                                raise ValueError(
                                    "Could not generate the selected seismic image."
                                )

                            # 3. GEOLOGICAL INTERPRETATION REQUEST

                            interpretation_query = f"""
            You are StrataMind, an expert petroleum geologist
            and seismic interpreter.

            Analyse ONLY the supplied selected seismic region.

            MAIN OBJECTIVE:
            Identify and assess possible hydrocarbon traps.

            STRUCTURAL INTERPRETATION:
            - Normal and reverse faults
            - Fault displacement
            - Anticlines and synclines
            - Fault closures
            - Structural highs
            - Horsts and tilted fault blocks

            STRATIGRAPHIC INTERPRETATION:
            - Onlap and downlap
            - Pinch-outs
            - Unconformities and truncation
            - Channels and channel-fill deposits
            - Reef or carbonate buildups
            - Stratigraphic termination

            SEISMIC INDICATORS:
            - Bright spots and dim spots
            - Flat spots
            - Amplitude anomalies
            - Chaotic and discontinuous reflectors
            - Reflection continuity and configuration

            TRAP ASSESSMENT:

            Classify each candidate as:
            - structural
            - stratigraphic
            - combination
            - not_established

            For each potential trap, explain:
            1. Directly observed seismic evidence.
            2. Geological interpretation.
            3. Potential trapping mechanism.
            4. Possible reservoir geometry.
            5. Possible seal mechanism.
            6. Alternative explanations.
            7. Additional validation required.

            ANNOTATION REQUIREMENTS:

            Return normalised coordinates from 0 to 1
            relative to the supplied cropped image.

            - x: left to right
            - y: top to bottom
            - line: faults and reflector trends
            - polygon: possible trap boundaries
            - box: localised features
            - point: individual anomalies

            IMPORTANT:
            - Annotate actual visible features only.
            - Do not invent faults, horizons or trap boundaries.
            - A fault alone does not establish a trap.
            - A bright spot alone does not prove hydrocarbons.
            - Distinguish observations from interpretations.
            - State uncertainty.
            - If no convincing trap exists, return an empty trap candidate list.

            Dataset: {seismic_data['filename']}
            Selected trace range:
            {st.session_state.selected_seismic_x_range}

            Selected time range:
            {st.session_state.selected_seismic_y_range} ms

            Return the structured JSON format required by StrataMind.
            """

                            # 4. RUN GEMMA AND ANNOTATE SELECTED IMAGE

                            result = analyse_seismic_image(
                                cropped_png,
                                user_query=textwrap.dedent(
                        interpretation_query
                    ).strip()
                            )

                            # 5. SAVE ANALYSIS INFORMATION

                            result["trace_range"] = (
                                st.session_state.selected_seismic_x_range
                            )

                            result["time_range"] = (
                                st.session_state.selected_seismic_y_range
                            )

                            result["original_filename"] = (
                                seismic_data["filename"]
                            )

                            result["analysis_scope"] = (
                                "Selected seismic region"
                            )

                            st.session_state.interpretation_filename = (
                                seismic_data["filename"]
                            )

                            st.session_state.seismic_interpretation_result = result

                        st.success(
                            "Selected seismic region interpreted successfully."
                        )

                    except Exception as error:

                        st.error(
                            f"Geological interpretation failed: {error}"
                        )

                        st.exception(error)

            # ========================================================
            # DISPLAY GEMMA INTERPRETATION
            # ========================================================

            result = st.session_state.get(
                "seismic_interpretation_result"
            )

            if (
                result is not None
                and st.session_state.get("interpretation_filename")
                == seismic_data["filename"]
            ):

                st.divider()

                st.markdown("### Geological Interpretation Results")

                # ----------------------------------------------------
                # 1. ANALYSIS INFORMATION
                # ----------------------------------------------------

                st.caption(
                    f"Dataset: {result.get('original_filename', 'Unknown')} | "
                    f"Scope: {result.get('analysis_scope', 'Seismic interpretation')}"
                )

                if result.get("trace_range") is not None:
                    st.write("**Selected trace range:**", result["trace_range"])

                if result.get("time_range") is not None:
                    st.write("**Selected time range (ms):**", result["time_range"])

                # ----------------------------------------------------
                # 2. ANNOTATED SEISMIC IMAGE
                # ----------------------------------------------------

                st.markdown("#### Annotated Seismic Region")

                st.image(
                    result["annotated_image"],
                    use_container_width=True
                )

                with st.expander("View Original Selected Seismic Region"):
                    st.image(
                        result["original_image"],
                        use_container_width=True
                    )

                # ----------------------------------------------------
                # 3. GEOLOGICAL SUMMARY
                # ----------------------------------------------------

                st.markdown("#### Geological Summary")

                st.write(
                    result.get(
                        "summary",
                        "No geological summary available."
                    )
                )

                # ----------------------------------------------------
                # 4. TRAP ASSESSMENT
                # ----------------------------------------------------

                st.divider()

                st.markdown("### Hydrocarbon Trap Assessment")

                trap_assessment = result.get(
                    "trap_assessment", {}
                )

                if not isinstance(trap_assessment, dict):
                    trap_assessment = {}

                overall_classification = trap_assessment.get(
                    "classification",
                    "not_established"
                )

                trap_candidates = result.get(
                    "trap_candidates",
                    trap_assessment.get("trap_candidates", [])
                )

                if not isinstance(trap_candidates, list):
                    trap_candidates = []

                trap_candidates = [
                    candidate for candidate in trap_candidates
                    if isinstance(candidate, dict)
                ]

                structural_count = sum(
                    1 for candidate in trap_candidates
                    if candidate.get("classification") == "structural"
                )

                stratigraphic_count = sum(
                    1 for candidate in trap_candidates
                    if candidate.get("classification") == "stratigraphic"
                )

                combination_count = sum(
                    1 for candidate in trap_candidates
                    if candidate.get("classification") == "combination"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Candidates",
                    len(trap_candidates)
                )

                col2.metric(
                    "Structural",
                    structural_count
                )

                col3.metric(
                    "Stratigraphic",
                    stratigraphic_count
                )

                col4.metric(
                    "Combination",
                    combination_count
                )

                st.write(
                    "**Overall classification:**",
                    str(overall_classification).replace(
                        "_", " "
                    ).title()
                )

                # ----------------------------------------------------
                # 5. INDIVIDUAL TRAP CANDIDATES
                # ----------------------------------------------------

                st.markdown("#### Identified Trap Candidates")

                if trap_candidates:

                    for index, candidate in enumerate(
                        trap_candidates,
                        start=1
                    ):

                        candidate_id = candidate.get(
                            "candidate_id",
                            f"TC-{index:03d}"
                        )

                        trap_type = str(
                            candidate.get("trap_type") or "Unknown trap type"
                        )

                        classification = str(
                            candidate.get("classification") or "not_established"
                        )

                        confidence = candidate.get(
                            "confidence",
                            0
                        )

                        try:
                            confidence_text = f"{float(confidence) * 100:.0f}%"
                        except (TypeError, ValueError):
                            confidence_text = "Not provided"

                        with st.expander(
                            f"{candidate_id} | "
                            f"{trap_type.replace('_', ' ').title()} | "
                            f"{classification.title()}",
                            expanded=True
                        ):

                            st.write(
                                "**Classification:**",
                                classification.replace(
                                    "_", " "
                                ).title()
                            )

                            st.write(
                                "**AI assessment confidence:**",
                                confidence_text
                            )

                            st.markdown("**Seismic Observations**")

                            observations = candidate.get(
                                "observations", []
                            )

                            if observations:
                                for observation in observations:
                                    st.write(f"- {observation}")
                            else:
                                st.write("No observations provided.")

                            st.markdown("**Geological Evidence**")

                            st.write(
                                candidate.get(
                                    "evidence",
                                    "No evidence provided."
                                )
                            )

                            st.markdown("**Geological Interpretation**")

                            st.write(
                                candidate.get(
                                    "interpretation",
                                    "No interpretation provided."
                                )
                            )

                            st.markdown("**Potential Trapping Mechanism**")

                            st.write(
                                candidate.get(
                                    "trapping_mechanism",
                                    "Not established."
                                )
                            )

                            st.markdown("**Possible Reservoir Geometry**")

                            st.write(
                                candidate.get(
                                    "reservoir_geometry",
                                    "Not established."
                                )
                            )

                            st.markdown("**Possible Seal Mechanism**")

                            st.write(
                                candidate.get(
                                    "seal_mechanism",
                                    "Not established."
                                )
                            )

                            st.markdown("**Alternative Explanations**")

                            alternatives = candidate.get(
                                "alternative_explanations", []
                            )

                            if alternatives:
                                for alternative in alternatives:
                                    st.write(f"- {alternative}")
                            else:
                                st.write("None provided.")

                            st.markdown("**Validation Required**")

                            validation = candidate.get(
                                "validation_required", []
                            )

                            if validation:
                                for item in validation:
                                    st.write(f"- {item}")
                            else:
                                st.write("No validation steps provided.")

                else:

                    st.info(
                        "Gemma did not identify any sufficiently supported "
                        "potential hydrocarbon trap candidates in this region."
                    )

                st.caption(
                    "AI confidence represents the model's assessment, "
                    "not a measured geological probability. "
                    "Trap candidates require further geological validation."
                )

                # ----------------------------------------------------
                # 6. INDIVIDUAL GEOLOGICAL FEATURES
                # ----------------------------------------------------

                st.divider()

                st.markdown("### Identified Geological Features")

                features = result.get("features", [])

                if not isinstance(features, list):
                    features = []

                features = [
                    feature for feature in features
                    if isinstance(feature, dict)
                ]

                if features:

                    for index, feature in enumerate(
                        features,
                        start=1
                    ):

                        feature_name = str(
                            feature.get(
                                "type",
                                feature.get(
                                    "name",
                                    feature.get("feature", f"Feature {index}")
                                )
                            )
                        )

                        with st.expander(
                            feature_name.replace(
                                "_", " "
                            ).title()
                        ):

                            st.write(
                                "**Category:**",
                                feature.get("category", "Unknown")
                            )

                            confidence = feature.get(
                                "confidence",
                                0
                            )

                            try:
                                confidence_text = (
                                    f"{float(confidence) * 100:.0f}%"
                                )
                            except (TypeError, ValueError):
                                confidence_text = "Not provided"

                            st.write(
                                "**AI assessment confidence:**",
                                confidence_text
                            )

                            st.write(
                                "**Evidence:**",
                                feature.get(
                                    "evidence",
                                    feature.get(
                                        "description",
                                        "Not provided."
                                    )
                                )
                            )

                            st.write(
                                "**Interpretation:**",
                                feature.get(
                                    "interpretation",
                                    "Not provided."
                                )
                            )

                else:

                    st.info(
                        "No individual geological features were returned."
                    )

                # ----------------------------------------------------
                # 7. DOWNLOAD ANNOTATED IMAGE
                # ----------------------------------------------------

                st.download_button(
                    "Download Annotated Seismic Region",
                    data=result["annotated_image"],
                    file_name="stratamind_annotated_seismic.png",
                    mime="image/png",
                    key="download_annotated_seismic"
                )

            elif result is not None:

                st.info(
                    "The previous interpretation belongs to a different "
                    "seismic dataset. Select an area and run a new analysis "
                    "to interpret the currently loaded section."
                )

            # ============================================================
            # SEG-Y METADATA AND PREVIEW DOWNLOAD
            # ============================================================

            with st.expander("View SEG-Y Metadata"):

                st.write(
                    "Sampling interval:",
                    seismic_data["sample_interval_us"],
                    "microseconds"
                )

                st.write(
                    "Amplitude matrix:",
                    seismic_data["amplitudes"].shape
                )

                st.write(
                    "Displayed trace range:",
                    f"{seismic_data['trace_axis'][0]} - "
                    f"{seismic_data['trace_axis'][-1]}"
                )

            # Build the CSV only when requested; it is large and would
            # otherwise be regenerated on every rerun.
            if st.checkbox(
                "Prepare seismic preview CSV for download",
                key="prepare_seismic_csv"
            ):

                st.download_button(
                    "Download Seismic Preview (CSV)",
                    data=pd.DataFrame(
                        seismic_data["amplitudes"].T
                    ).to_csv(index=False),
                    file_name="seismic_preview.csv",
                    mime="text/csv",
                    key="download_seismic_preview"
                )

        else:

            st.info(
                "Select a dataset or upload a SEG-Y file "
                "to display the seismic section."
            )

# ============================================================
# 14. FAR-RIGHT PANEL: ORIGINAL GEMMA AI CHAT
# ============================================================

with chat_col:

    with st.container(border=True):

        st.markdown("### 🤖 StrataMind AI")

        st.caption(
            "Your geological conversational assistant, "
            "powered by Gemma."
        )

        # ----------------------------------------------------
        # CONVERSATION HISTORY
        # ----------------------------------------------------

        chat_history = st.container(height=430)

        with chat_history:

            if not conversation["messages"]:

                st.info(
                    "Welcome to StrataMind! Upload your "
                    "geological data or ask a question."
                )

            for message in conversation["messages"]:

                with st.chat_message(message["role"]):

                    if message.get("text"):

                        st.markdown(message["text"])

                    if message["role"] == "user":

                        for attachment in message.get(
                            "attachments", []
                        ):

                            st.caption(
                                f"📎 {attachment['name']}"
                            )

        st.divider()

        # ----------------------------------------------------
        # OPTIONAL WELL-TO-SEISMIC CORRELATION
        # ----------------------------------------------------

        checkshot_file = None

        well_loaded = (
            st.session_state.get("well_log_df") is not None
        )

        seismic_loaded = (
            st.session_state.get("seismic_section") is not None
        )

        if well_loaded and seismic_loaded:

            st.markdown("#### 🔗 Well-to-Seismic Correlation")

            st.caption(
                "Connect your loaded well-log dataset "
                "to a selected trace on the seismic section."
            )

            enable_correlation = st.checkbox(
                "I want to correlate this well with seismic",
                key="enable_well_seismic_correlation"
            )

            if enable_correlation:

                seismic_for_correlation = (
                    st.session_state.seismic_section
                )

                available_traces = (
                    seismic_for_correlation["trace_axis"].tolist()
                )

                middle_index = len(available_traces) // 2

                st.info(
                    "Select the seismic trace where you believe "
                    "the well is located. This is a user-selected "
                    "position and is not automatically verified."
                )

                selected_correlation_trace = st.selectbox(
                    "Select seismic trace for the well",
                    options=available_traces,
                    index=middle_index,
                    key="correlation_trace_selection"
                )

                st.markdown("**Optional supporting data**")

                checkshot_file = st.file_uploader(
                    "Upload checkshot data",
                    type=["txt", "csv", "dat"],
                    accept_multiple_files=False,
                    key="checkshot_upload",
                    help=(
                        "Expected columns: TVDSS in metres "
                        "and TWT in milliseconds."
                    )
                )

                st.caption(
                    "Your LAS depth is MDSS, whereas the "
                    "checkshot depth is TVDSS. Without a "
                    "deviation survey, these depth references "
                    "cannot be assumed equivalent."
                )

                if st.button(
                    "🔗 Generate Seismic Correlation Overlay",
                    use_container_width=True,
                    key="generate_correlation_overlay"
                ):

                    checkshot_df = None
                    checkshot_name = None
                    checkshot_ok = True

                    if checkshot_file is not None:

                        try:

                            checkshot_result = read_checkshot_txt(
                                checkshot_file.getvalue(),
                                filename=checkshot_file.name
                            )

                            checkshot_df = (
                                checkshot_result["data"]
                            )

                            checkshot_name = (
                                checkshot_result["filename"]
                            )

                        except Exception as error:

                            st.error(
                                f"Unable to read checkshot data: {error}"
                            )

                            checkshot_ok = False

                    if checkshot_ok:

                        st.session_state.well_seismic_correlation = {
                            "trace": selected_correlation_trace,
                            "checkshot_data": checkshot_df,
                            "checkshot_filename": checkshot_name,
                            "well_filename": (
                                st.session_state.get(
                                    "well_log_filename",
                                    "Unknown well"
                                )
                            ),
                            "seismic_filename": (
                                seismic_for_correlation["filename"]
                            )
                        }

                        st.success(
                            "Correlation overlay generated. "
                            "View the annotations in the 2D Seismic Explorer."
                        )

                        st.rerun()

                if st.session_state.get(
                    "well_seismic_correlation"
                ) is not None:

                    st.caption(
                        "A correlation overlay is currently active."
                    )

                    if st.button(
                        "Remove Correlation Overlay",
                        use_container_width=True,
                        key="remove_correlation_overlay"
                    ):

                        st.session_state.well_seismic_correlation = None

                        st.rerun()

        # ----------------------------------------------------
        # QUICK GEOLOGICAL QUESTIONS
        # ----------------------------------------------------

        st.markdown("**Suggested questions**")

        prompt_col1, prompt_col2 = st.columns(2)

        quick_question = None

        with prompt_col1:

            if st.button(
                "🔍 Identify traps",
                key="prompt_traps",
                use_container_width=True
            ):

                quick_question = (
                    "Analyse the loaded seismic and well-log data. "
                    "What possible structural and stratigraphic traps "
                    "can you identify? Explain the evidence."
                )

            if st.button(
                "📐 Structural features",
                key="prompt_structures",
                use_container_width=True
            ):

                quick_question = (
                    "Identify possible faults, anticlines, synclines, "
                    "structural highs and fault displacement in the "
                    "loaded seismic section."
                )

        with prompt_col2:

            if st.button(
                "🌊 Seismic features",
                key="prompt_features",
                use_container_width=True
            ):

                quick_question = (
                    "Describe the visible seismic reflection patterns, "
                    "including continuity, amplitude anomalies, "
                    "chaotic reflections and possible channels."
                )

            if st.button(
                "🪨 Stratigraphic features",
                key="prompt_stratigraphy",
                use_container_width=True
            ):

                quick_question = (
                    "Investigate possible onlap, downlap, toplap, "
                    "unconformities, pinch-outs and stratigraphic traps."
                )

        # ----------------------------------------------------
        # ORIGINAL CHAT INPUT
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask StrataMind about your geological data..."
        )

        if quick_question is not None:
            question = quick_question

# ============================================================
# 15. PROCESS USER QUESTION
# ============================================================

if question:

    # Process uploaded files
    attachments = process_uploaded_files(
        uploaded_files or [],
        selected_well=selected_well
    )

    # --------------------------------------------------------
    # ADD LOADED WELL-LOG INFORMATION TO GEMMA
    # --------------------------------------------------------

    well_df = st.session_state.get("well_log_df")

    if well_df is not None:

        detected = st.session_state.get(
            "detected_curves", {}
        )

        depth_column = detected.get("DEPTH", "DEPTH")

        available_columns = list(dict.fromkeys(
            [depth_column] + [
                column for column in detected.values()
                if column in well_df.columns
            ]
        ))

        available_columns = [
            column for column in available_columns
            if column in well_df.columns
        ]

        well_sample = well_df[available_columns].head(25)

        well_context = (
            "LOADED WELL-LOG INFORMATION:\n"
            f"Filename: "
            f"{st.session_state.get('well_log_filename', 'Unknown')}\n"
            "Depth reference: MDSS, in metres, as supplied by the well-log dataset.\n"
            f"Available curves: {list(detected.keys())}\n"
            f"Number of depth samples: {len(well_df)}\n\n"
            "Sample of the loaded well-log data:\n"
            + well_sample.to_string(index=False)
            + "\n\n"
            "Interpret the well-log information using the "
            "available curves. Do not assume that MD is "
            "equivalent to true vertical depth (TVD)."
        )

        attachments.append({
            "name": "Loaded well-log information",
            "kind": "text",
            "content": well_context
        })

    # --------------------------------------------------------
    # READ AND ADD OPTIONAL CHECKSHOT DATA
    # --------------------------------------------------------

    if checkshot_file is not None:

        try:

            checkshot_result = read_checkshot_txt(
                checkshot_file.getvalue(),
                filename=checkshot_file.name
            )

            checkshot_df = checkshot_result["data"]

            checkshot_context = (
                "VALIDATED CHECKSHOT INFORMATION:\n\n"
                f"Filename: {checkshot_result['filename']}\n"
                f"Well name: {checkshot_result['well_name']}\n"
                f"Depth reference: TVDSS (metres)\n"
                f"Time reference: TWT (milliseconds)\n"
                f"Number of measurements: "
                f"{checkshot_result['number_of_points']}\n\n"
                "CHECKSHOT MEASUREMENTS:\n"
                + checkshot_df.to_string(index=False)
                + "\n\n"
                "IMPORTANT CORRELATION LIMITATIONS:\n"
                "- Checkshot depth is TVDSS, not MDSS.\n"
                "- Do not directly equate MDSS with TVDSS.\n"
                "- No deviation survey is available.\n"
                "- Do not claim an exact well-to-trace match.\n"
                "- Use TWT as provided. Do not double it again.\n"
                "- Identify any additional information needed "
                "for a reliable well-to-seismic tie."
            )

            attachments.append({
                "name": checkshot_file.name,
                "kind": "text",
                "content": checkshot_context
            })

            st.success(
                "Checkshot data successfully read and validated."
            )

        except Exception as e:

            st.error(
                f"Unable to read checkshot data: {e}"
            )

    # --------------------------------------------------------
    # ADD LOADED HPC / UPLOADED SEISMIC METADATA TO GEMMA
    # --------------------------------------------------------

    seismic_data = st.session_state.get(
        "seismic_section"
    )

    if seismic_data is not None:

        amplitudes = seismic_data["amplitudes"]

        valid_amplitudes = amplitudes[
            np.isfinite(amplitudes)
        ]

        if valid_amplitudes.size > 0:

            seismic_context = (
                "Currently loaded 2D seismic dataset:\n"
                f"Filename: {seismic_data['filename']}\n"
                f"Total traces: {seismic_data['trace_count']}\n"
                f"Samples per trace: {seismic_data['sample_count']}\n"
                f"Displayed traces: {seismic_data['displayed_traces']}\n"
                f"Sampling interval: "
                f"{seismic_data['sample_interval_us']} microseconds\n"
                f"Preview amplitude matrix: {amplitudes.shape}\n"
                f"Minimum amplitude: "
                f"{np.min(valid_amplitudes):.4f}\n"
                f"Maximum amplitude: "
                f"{np.max(valid_amplitudes):.4f}\n"
                f"Mean amplitude: "
                f"{np.mean(valid_amplitudes):.4f}\n"
                f"RMS amplitude: "
                f"{np.sqrt(np.mean(valid_amplitudes ** 2)):.4f}\n\n"
                "Important: These are preview statistics. "
                "They do not represent a complete geological "
                "interpretation of the seismic dataset."
            )

            attachments.append({
                "name": "Loaded seismic dataset metadata",
                "kind": "text",
                "content": seismic_context
            })

            # ----------------------------------------------------
            # SEND ACTUAL SEISMIC IMAGE TO GEMMA
            # ----------------------------------------------------

            try:

                seismic_image_bytes = create_seismic_ai_image(
                    seismic_data
                )

                attachments.append({
                    "name": "Loaded seismic section image",
                    "kind": "image",
                    "image_bytes": seismic_image_bytes,
                    "mime_type": "image/png"
                })

            except Exception as image_error:

                st.warning(
                    f"Unable to create seismic image for AI analysis: "
                    f"{image_error}"
                )

    # --------------------------------------------------------
    # CREATE USER MESSAGE
    # --------------------------------------------------------

    user_message = {
        "role": "user",
        "text": question,
        "attachments": attachments
    }

    conversation["messages"].append(user_message)

    # Set conversation title
    if len(conversation["messages"]) == 1:

        conversation["title"] = (
            question[:35] + "..."
            if len(question) > 35
            else question
        )

    # --------------------------------------------------------
    # GENERATE ORIGINAL GEMMA RESPONSE
    # --------------------------------------------------------

    with st.spinner(
        "StrataMind is analysing your geological data..."
    ):

        try:

            answer = generate_ai_response(
                conversation["messages"],
                selected_well
            )

        except Exception as e:

            answer = (
                "⚠️ Gemma could not complete the request.\n\n"
                f"Error: `{e}`\n\n"
                "Please check the model connection and "
                "ensure the vLLM server is running."
            )

    # Save assistant response
    conversation["messages"].append({
        "role": "assistant",
        "text": answer
    })

    st.rerun()

