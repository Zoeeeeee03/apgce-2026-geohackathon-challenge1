import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import openai
from well_knowledge import WELL_DATABASE

# 1. Page Config
st.set_page_config(page_title="APGCE Exploration Copilot", page_icon="🌍", layout="wide")

st.title("🌍 APGCE GeoHackathon 2026 — Subsurface Exploration Copilot")
st.caption("Agentic AI Prospect Ranking & Pre-Stack AVO Petrophysical Evaluation System")

# Connect to Local HPC vLLM Gemma 4 Endpoint
client = openai.OpenAI(
    base_url="http://172.28.34.1:8000/v1",
    api_key="token-not-needed"
)

# 2. Left Sidebar Navigation
st.sidebar.title("⚙️ Control Panel")
navigation_mode = st.sidebar.radio(
    "Select Workflow Mode:",
    ["📊 Single Well Deep-Dive", "🏆 Multi-Well Prospect Ranking"]
)

st.sidebar.divider()

# Helper Function: 4-Track Well Log Plotter (Cleaned without VP)
def plot_4_track_well_logs(well_name, depth_range=None):
    las_file_path = f"/data/epic20/APGCE/input_data/challenge1_challenge2/well_data/las/{well_name.lower()}.las"
    
    if not os.path.exists(las_file_path):
        st.error(f"LAS file for well {well_name.upper()} not found.")
        return None, None

    las = lasio.read(las_file_path)
    df = las.df()

    # Calculate Hydrocarbon Saturation (Sh = 1 - SWT)
    if 'SWT' in df.columns:
        df['SH'] = 1.0 - df['SWT']
    else:
        df['SH'] = 0.0

    # Filter depth range if zoomed
    if depth_range:
        df_sub = df[(df.index >= depth_range[0]) & (df.index <= depth_range[1])]
    else:
        df_sub = df.copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True, dpi=100)
    fig.suptitle(f"4-Track Wireline & Evidence Logs — Well {well_name.upper()}", fontsize=14, fontweight='bold')

    # Track 1: Lithology (VSH)
    if 'VSH' in df_sub.columns:
        axes[0].plot(df_sub['VSH'], df_sub.index, color='green', linewidth=0.9)
        axes[0].set_title("Track 1: Lithology\n(VSH)", fontsize=10)
        axes[0].set_xlim(0, 1)
        axes[0].axvline(0.3, color='orange', linestyle='--', label='Vsh < 0.3')
        axes[0].fill_betweenx(df_sub.index, 0, df_sub['VSH'], where=(df_sub['VSH'] < 0.3), color='yellow', alpha=0.5)
        axes[0].legend(loc='lower right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "VSH Not in LAS", ha='center', va='center')

    # Track 2: Porosity (PHIT)
    if 'PHIT' in df_sub.columns:
        axes[1].plot(df_sub['PHIT'], df_sub.index, color='red', linewidth=0.9)
        axes[1].set_title("Track 2: Porosity\n(PHIT)", fontsize=10)
        axes[1].set_xlim(0, 0.4)
        axes[1].axvline(0.15, color='darkred', linestyle='--', label='Phi > 0.15')
        axes[1].fill_betweenx(df_sub.index, 0.15, df_sub['PHIT'], where=(df_sub['PHIT'] > 0.15), color='coral', alpha=0.4)
        axes[1].legend(loc='lower right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "PHIT Not in LAS", ha='center', va='center')

    # Track 3: Hydrocarbon Saturation (SH = 1 - SWT)
    if 'SH' in df_sub.columns:
        axes[2].plot(df_sub['SH'], df_sub.index, color='darkgreen', linewidth=0.9)
        axes[2].set_title("Track 3: Fluid Saturation\n(Hydrocarbon Sh)", fontsize=10)
        axes[2].set_xlim(0, 1)
        axes[2].axvline(0.5, color='black', linestyle='--', label='Sh > 0.5')
        axes[2].fill_betweenx(df_sub.index, 0.5, df_sub['SH'], where=(df_sub['SH'] > 0.5), color='gold', alpha=0.6)
        axes[2].legend(loc='lower right', fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "SH Not in LAS", ha='center', va='center')

    # Track 4: Direct Pay Flags (PS_GIP & PS_OIP)
    axes[3].set_title("Track 4: Pay Flags\n(PS_GIP / PS_OIP)", fontsize=10)
    has_flag = False
    if 'PS_GIP' in df_sub.columns:
        axes[3].plot(df_sub['PS_GIP'], df_sub.index, color='blue', linewidth=1.0, label='Gas Flag (GIP)')
        has_flag = True
    if 'PS_OIP' in df_sub.columns:
        axes[3].plot(df_sub['PS_OIP'], df_sub.index, color='darkgreen', linestyle='--', linewidth=1.0, label='Oil Flag (OIP)')
        has_flag = True
    if has_flag:
        axes[3].legend(loc='lower right', fontsize=8)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, "Pay Flags\nNot in LAS", ha='center', va='center')

    axes[0].set_ylabel('Depth (m)', fontsize=11)
    axes[0].invert_yaxis()
    plt.tight_layout()
    
    return fig, df_sub


# ==========================================
# MODE 1: SINGLE WELL DEEP-DIVE
# ==========================================
if navigation_mode == "📊 Single Well Deep-Dive":
    st.sidebar.subheader("🎯 Select Target Well")
    selected_well = st.sidebar.radio(
        "Choose from Field Wells:",
        options=list(WELL_DATABASE.keys()),
        format_func=lambda x: f"Well {WELL_DATABASE[x]['well_name']}"
    )
    
    well_info = WELL_DATABASE[selected_well]
    
    st.header(f"Well Overview: {well_info['well_name']}")
    
    # OVERALL FIELD SUMMARY AT TOP (FIXED TRUNCATION)
    with st.container(border=True):
        st.subheader("🌐 Overall Field Summary (Entire Logged Column)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Logged Samples", f"{well_info['total_logged_samples']:,}")
        m2.metric("Total Net Pay Samples", f"{well_info['net_pay_samples']}")
        m3.metric("Avg Hydrocarbon Saturation (Sh)", well_info["pay_zone_properties"].get("hydrocarbon_saturation", "62.8%"))
        
        # Standalone row to display full primary pay depth range without truncation
        st.info(f"📍 **Overall Field Primary Pay Depth Range:** `{well_info['pay_zone_depth']}`")

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.subheader("📋 Subsurface Characterization")
            st.markdown(f"• **Primary Seal:** {well_info.get('primary_seal', 'Regional Marine Shale')}")
            st.markdown(f"• **Seismic Correlation:** {well_info.get('seismic_inversion_correlation', 'Correlates with Low Zp Volume')}")
            st.markdown(f"• **Recommended Action:** {well_info.get('recommended_next_action', 'Evaluate lateral sand continuity')}")

    with col_right:
        with st.container(border=True):
            st.subheader("🎯 Pre-Stack AVO & Elastic Inversion")
            avo_data = well_info.get("avo_inversion", {
                "vp_vs_ratio": "1.62",
                "p_impedance": "6,200 (m/s*g/cc)",
                "s_impedance": "3,800 (m/s*g/cc)",
                "avo_class": "Class III (Bright Spot)"
            })
            st.markdown(f"• **AVO Anomaly Class:** **{avo_data.get('avo_class', 'Class III (Bright Spot)')}**")
            st.markdown(f"• **Vp/Vs Ratio:** **{avo_data.get('vp_vs_ratio', '1.62')}**")
            st.markdown(f"• **P-Impedance ($Z_p$):** **{avo_data.get('p_impedance', '6,200')}** | **S-Impedance ($Z_s$):** **{avo_data.get('s_impedance', '3,800')}**")

    st.divider()

    # BOTTOM INTERACTIVE WIRELINE & DYNAMIC DEPTH RANGE EVALUATION
    st.subheader("📈 Interactive Wireline Logs & Specific Depth Range Analysis")
    
    # Geological Justification Note
    with st.expander("📖 Why is an interval classified as a Pay Zone? (Domain Guidance)", expanded=False):
        st.write("""
        An interval is confirmed as a **Net Pay Reservoir Zone** when petrophysical measurements concurrently satisfy three cutoff criteria:
        1. **Low Shale Volume ($V_{shale} < 0.30$):** Confirms clean sandstone instead of impermeable mudstone/shale.
        2. **High Porosity ($PHIT > 0.15$ or $15\%$):** Confirms sufficient fluid storage space within rock pores.
        3. **High Hydrocarbon Saturation ($S_h > 0.50$):** Confirms the presence of gas/oil ($S_h = 1 - S_w$) instead of formation water.
        """)

    las_path = f"/data/epic20/APGCE/input_data/challenge1_challenge2/well_data/las/{selected_well.lower()}.las"
    if os.path.exists(las_path):
        las_temp = lasio.read(las_path)
        min_d, max_d = float(las_temp.depth_m[0]), float(las_temp.depth_m[-1])
        st.markdown(f"**Target Pay Depth Bounds for {selected_well.upper()}:** `{well_info['pay_zone_depth']}`")
        zoom_range = st.slider("🔍 Zoom Depth Range (m):", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=1.0)
    else:
        zoom_range = None

    fig, df_sub = plot_4_track_well_logs(selected_well, depth_range=zoom_range)
    if fig:
        st.pyplot(fig)

    # INTERVAL-SPECIFIC RE-CALCULATED METRICS
    if df_sub is not None and not df_sub.empty and zoom_range:
        st.markdown(f"### 🎯 Re-Calculated Properties for Selected Range (`{zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m`)")
        
        avg_vsh = df_sub['VSH'].mean() if 'VSH' in df_sub.columns else np.nan
        avg_phit = df_sub['PHIT'].mean() if 'PHIT' in df_sub.columns else np.nan
        avg_sh = df_sub['SH'].mean() if 'SH' in df_sub.columns else np.nan
        
        is_pay = (avg_vsh < 0.3) and (avg_phit > 0.15) and (avg_sh > 0.5)

        with st.container(border=True):
            d1, d2, d3 = st.columns(3)
            d1.metric("Interval Avg Shale Vol (VSH)", f"{avg_vsh*100:.1f}%" if not np.isnan(avg_vsh) else "N/A")
            d2.metric("Interval Avg Porosity (PHIT)", f"{avg_phit*100:.1f}%" if not np.isnan(avg_phit) else "N/A")
            d3.metric("Interval Avg HC Saturation (Sh)", f"{avg_sh*100:.1f}%" if not np.isnan(avg_sh) else "N/A")

            st.markdown("**Interval Interpretation:**")
            if is_pay:
                st.success(f"✅ **Commercial Pay Confirmed for {zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m:** Low shale volume (<30%), high porosity (>15%), and strong hydrocarbon saturation (>50%).")
            else:
                st.info(f"ℹ️ **Non-Pay / Tight / Water-Bearing Interval for {zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m:** Does not meet full pay cutoffs. Average parameters indicate non-productive shale or brine-bearing matrix.")

    st.divider()

    # Pop-Up AI Agent
    with st.popover("💬 Click to Open Exploration Copilot AI Assistant", use_container_width=True):
        st.markdown(f"### Exploration Copilot — Well {well_info['well_name']} Reasoning")
        user_input = st.text_input("Ask Copilot a question regarding selected depth or properties:")
        if user_input:
            system_prompt = f"You are Exploration Copilot evaluating Well {well_info['well_name']}."
            resp = client.chat.completions.create(
                model="google/gemma-4-26B-A4B-it",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
                temperature=0.2
            )
            st.markdown(resp.choices[0].message.content)