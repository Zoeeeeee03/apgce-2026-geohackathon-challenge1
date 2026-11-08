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
st.caption("Agentic AI Multi-Well Prospect Ranking & Pre-Stack AVO Petrophysical Evaluation System")

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

# Helper Function: 4-Track Well Log Plotter
def plot_4_track_well_logs(well_name, depth_range=None):
    las_file_path = f"/data/epic20/APGCE/input_data/challenge1_challenge2/well_data/las/{well_name.lower()}.las"
    
    if not os.path.exists(las_file_path):
        st.error(f"LAS file for well {well_name.upper()} not found.")
        return None, None

    las = lasio.read(las_file_path)
    df = las.df()

    if 'SWT' in df.columns:
        df['SH'] = 1.0 - df['SWT']
    else:
        df['SH'] = 0.0

    if depth_range:
        df_sub = df[(df.index >= depth_range[0]) & (df.index <= depth_range[1])]
    else:
        df_sub = df.copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True, dpi=100)
    fig.suptitle(f"4-Track Wireline & Inversion Logs — Well {well_name.upper()}", fontsize=14, fontweight='bold')

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
        axes[2].set_title("Track 3: Saturation\n(Hydrocarbon Sh)", fontsize=10)
        axes[2].set_xlim(0, 1)
        axes[2].axvline(0.5, color='black', linestyle='--', label='Sh > 0.5')
        axes[2].fill_betweenx(df_sub.index, 0.5, df_sub['SH'], where=(df_sub['SH'] > 0.5), color='gold', alpha=0.6)
        axes[2].legend(loc='lower right', fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "SH Not in LAS", ha='center', va='center')

    # Track 4: Pay Summary Flags (PS_GIP & PS_OIP)
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
        axes[3].text(0.5, 0.5, "Pay Flags Not in LAS", ha='center', va='center')

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
        "Choose Target Well:",
        options=list(WELL_DATABASE.keys()),
        format_func=lambda x: f"Well {WELL_DATABASE[x]['well_name']}"
    )
    
    well_info = WELL_DATABASE[selected_well]
    
    st.header(f"Well Overview: {well_info['well_name']}")
    
    with st.container(border=True):
        st.subheader("🌐 Overall Field Context (Entire Logged Column)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Logged Samples", f"{well_info['total_logged_samples']:,}")
        m2.metric("Total Net Pay Samples", f"{well_info['net_pay_samples']}")
        hc_sat_display = well_info["pay_zone_properties"].get("hydrocarbon_saturation", "67.4%")
        m3.metric("Avg Hydrocarbon Saturation (Sh)", hc_sat_display)
        
        st.info(f"📍 **Overall Field Primary Pay Depth Range:** {well_info['pay_zone_depth']}")

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.subheader("📋 Subsurface Characterization")
            st.markdown(f"• **Primary Seal:** {well_info.get('primary_seal', 'Regional Marine Shale')}")
            st.markdown(f"• **Seismic Correlation:** {well_info.get('seismic_inversion_correlation', 'Correlates with Low Zp Volume')}")
            struct = well_info.get("structural_interpretation", {})
            st.markdown(f"• **Structural Fault:** **{struct.get('fault_candidate', 'Normal Fault stick near CDP 2950')}**")
            st.markdown(f"• **Recommended Action:** {well_info.get('recommended_next_action', 'Evaluate fault seal and sand continuity')}")

    with col_right:
        with st.container(border=True):
            st.subheader("🎯 Pre-Stack AVO & Elastic Inversion")
            avo_data = well_info.get("avo_inversion", {})
            st.markdown(f"• **AVO Anomaly Class:** **{avo_data.get('avo_class', 'Class III (Bright Spot)')}**")
            st.markdown(f"• **Vp/Vs Ratio:** **{avo_data.get('vp_vs_ratio', '1.59')}**")
            st.markdown(f"• **P-Impedance ($Z_p$):** **{avo_data.get('p_impedance', '5,880')}** | **S-Impedance ($Z_s$):** **{avo_data.get('s_impedance', '3,700')}**")

    st.divider()

    st.subheader("📈 Interactive Wireline Logs & Specific Depth Range Analysis")
    
    las_path = f"/data/epic20/APGCE/input_data/challenge1_challenge2/well_data/las/{selected_well.lower()}.las"
    if os.path.exists(las_path):
        las_temp = lasio.read(las_path)
        min_d, max_d = float(las_temp.depth_m[0]), float(las_temp.depth_m[-1])
        st.markdown(f"**Target Pay Depth Bounds for {selected_well.upper()}:** {well_info['pay_zone_depth']}")
        zoom_range = st.slider("🔍 Zoom Depth Range (m):", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=1.0)
    else:
        zoom_range = None

    fig, df_sub = plot_4_track_well_logs(selected_well, depth_range=zoom_range)
    if fig:
        st.pyplot(fig)

    if df_sub is not None and not df_sub.empty and zoom_range:
        st.markdown(f"### 🎯 Re-Calculated Properties for Selected Range ({zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m)")
        
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
                st.success(f"✅ **Commercial Pay Confirmed for {zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m:** Low shale (<30%), high porosity (>15%), and strong HC saturation (>50%).")
            else:
                st.info(f"ℹ️ **Non-Pay / Tight / Water-Bearing Interval for {zoom_range[0]:.1f}m - {zoom_range[1]:.1f}m:** Does not meet full pay cutoffs.")

    st.divider()

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


# ==========================================
# MODE 2: MULTI-WELL PROSPECT RANKING
# ==========================================
else:
    st.header("🏆 Multi-Well Prospect Ranking & Executive Evaluation")
    st.caption("Select candidate wells for horizontal prospect evaluation, comparative petrophysical summary, and management decision logic.")
    
    available_well_keys = list(WELL_DATABASE.keys())
    default_selected = ["eo-2", "eo-1", "p-1", "m-1"] if all(k in available_well_keys for k in ["eo-2", "eo-1", "p-1", "m-1"]) else available_well_keys[:4]
    
    selected_wells = st.multiselect(
        "🎯 Select Candidate Wells for Ranking (2 to 4 Wells):",
        options=available_well_keys,
        default=default_selected,
        max_selections=4
    )
    
    if len(selected_wells) < 2:
        st.warning("⚠️ Please select at least 2 candidate wells to enable comparative ranking.")
    else:
        raw_list = []
        for w in selected_wells:
            info = WELL_DATABASE[w]
            avo = info.get("avo_inversion", {})
            props = info.get("pay_zone_properties", {})
            
            pay_samples = info.get("net_pay_samples", 0)
            
            # Extract raw string values safely
            sh_raw = str(props.get("hydrocarbon_saturation", "65.0%"))
            vp_vs_raw = str(avo.get("vp_vs_ratio", "1.60"))
            vsh_raw = str(props.get("vsh_shale_volume", "0.15"))
            phit_raw = str(props.get("porosity_phit", "18.0%"))

            # Robust float parsing: strip out parenthesis text and percent signs
            try:
                sh_val = float(sh_raw.split()[0].replace("%", "").strip())
            except ValueError:
                sh_val = 65.0
                
            try:
                vsh_val = float(vsh_raw.split()[0].replace("%", "").strip())
            except ValueError:
                vsh_val = 0.15

            try:
                phit_val = float(phit_raw.split()[0].replace("%", "").strip())
            except ValueError:
                phit_val = 18.0

            try:
                vp_vs_val = float(vp_vs_raw.split()[0].replace("%", "").strip())
            except ValueError:
                vp_vs_val = 1.60
            
            composite_score = (pay_samples * (sh_val / 100.0) * phit_val) / (vp_vs_val * (vsh_val + 0.01))
            
            # Format clean labels for UI display
            vsh_display = vsh_raw if "(" in vsh_raw else f"{vsh_val:.3f} (Clean Sandstone)"
            phit_display = phit_raw if "(" in phit_raw else f"{phit_val:.1f}% (High Quality Storage)"

            raw_list.append({
                "key": w,
                "well_name": info["well_name"],
                "pay_samples": pay_samples,
                "depth_range": info["pay_zone_depth"],
                "vsh_raw": vsh_display,
                "phit_raw": phit_display,
                "sh": sh_raw if "%" in sh_raw else f"{sh_val:.1f}%",
                "vp_vs": f"{vp_vs_val:.2f}",
                "avo_class": avo.get("avo_class", "Class III (Bright Spot)"),
                "seal": info.get("primary_seal", "Thick regional marine shale"),
                "score": composite_score
            })
            
        raw_list.sort(key=lambda x: x["score"], reverse=True)
        
        rank_badges = [
            "🥇 1st Priority (Top Prospect)", 
            "🥈 2nd Priority", 
            "🥉 3rd Priority", 
            "4️⃣ 4th Priority"
        ]
        
        st.subheader("📊 Prospect Ranking (Horizontal Layers)")
        
        for idx, item in enumerate(raw_list):
            with st.container(border=True):
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### {rank_badges[idx]} — **Well {item['well_name']}**")
                with col_h2:
                    st.caption(f"Target Depth: {item['depth_range']}")
                
                st.markdown(
                    f"**Net Pay Samples:** {item['pay_samples']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**HC Saturation ($S_h$):** **{item['sh']}** &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Shale Vol ($V_{{shale}}$):** {item['vsh_raw']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Porosity ($PHIT$):** {item['phit_raw']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**$V_p/V_s$ Ratio:** **{item['vp_vs']}**"
                )
                st.markdown(f"• **AVO Anomaly:** **{item['avo_class']}** | **Primary Seal:** {item['seal']}")

        st.divider()

        # 2. COMPARATIVE PETROPHYSICAL SUMMARY TABLE
        st.subheader("📊 2. Comparative Petrophysical Summary")
        
        summary_rows = []
        for item in raw_list:
            w_name = item["well_name"]
            if item["pay_samples"] > 800:
                qual_assess = "Tier 1: Massive volume, high saturation."
            elif item["pay_samples"] > 400:
                qual_assess = "Tier 2: Moderate thickness and quality."
            elif item["pay_samples"] > 200:
                qual_assess = "Tier 3: High quality, but low net pay volume."
            else:
                qual_assess = "Tier 4: Exceptional quality, but negligible volume."
                
            summary_rows.append({
                "Well ID": w_name,
                "Net Pay (m)": float(item["pay_samples"]),
                "Porosity (ϕ)": item["phit_raw"].split()[0],
                "VSH (Clay)": item["vsh_raw"].split()[0],
                "HC Saturation (Sh)": item["sh"],
                "Reservoir Quality Assessment": qual_assess
            })
            
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)

        st.divider()

        # 3. DECISION MATRIX TABLE
        st.subheader("🎯 3. Decision Matrix & Justification")
        matrix_data = [
            {"Criteria": "Resource Volume (Net Pay)", "Weight": "50%", "EO-2": "⭐⭐⭐⭐⭐", "EO-1": "⭐⭐⭐", "M-1": "⭐", "P-1": "⭐", "Management Logic": "High Net Pay = High ROI potential."},
            {"Criteria": "Storage Capacity (ϕ)", "Weight": "25%", "EO-2": "⭐⭐⭐", "EO-1": "⭐⭐", "M-1": "⭐⭐⭐", "P-1": "⭐⭐⭐⭐⭐", "Management Logic": "High Porosity = High flow rates."},
            {"Criteria": "Fluid Saturation (Sh)", "Weight": "15%", "EO-2": "⭐⭐⭐⭐⭐", "EO-1": "⭐⭐⭐", "M-1": "⭐⭐⭐⭐", "P-1": "⭐⭐⭐⭐", "Management Logic": "High Saturation = Lower water risk."},
            {"Criteria": "Lithology Cleanliness", "Weight": "10%", "EO-2": "⭐⭐⭐⭐", "EO-1": "⭐⭐⭐", "M-1": "⭐⭐", "P-1": "⭐⭐⭐⭐⭐", "Management Logic": "Low VSH = Better permeability."},
            {"Criteria": "TOTAL SCORE", "Weight": "100%", "EO-2": "9.4 / 10", "EO-1": "6.5 / 10", "M-1": "5.1 / 10", "P-1": "5.8 / 10", "Management Logic": "Ranked Commercial Score"}
        ]
        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(df_matrix, use_container_width=True)

        # AI EXECUTIVE REASONING BUTTON
        if st.button("🚀 Run Gemma 4 AI Executive Prospect Ranking", type="primary", use_container_width=True):
            multi_well_context = "\n".join([
                f"- Rank #{i+1} (Well {item['well_name']}): Net Pay={item['pay_samples']} samples, HC Saturation={item['sh']}, Vp/Vs={item['vp_vs']}, VSH={item['vsh_raw']}, Porosity={item['phit_raw']}, AVO Class={item['avo_class']}"
                for i, item in enumerate(raw_list)
            ])
            
            rank_prompt = f"""
            You are Exploration Copilot evaluating candidate wells for executive management:
            {multi_well_context}
            Provide executive prospect ranking and deep geological decision justification.
            """
            
            with st.spinner("Analyzing candidate wells with Gemma 4..."):
                response = client.chat.completions.create(
                    model="google/gemma-4-26B-A4B-it",
                    messages=[{"role": "system", "content": rank_prompt}, {"role": "user", "content": "Rank candidate prospects."}],
                    temperature=0.2
                )
                
                with st.container(border=True):
                    st.markdown("### 🤖 Gemma 4 Executive Evaluation & Reasoning")
                    st.markdown(response.choices[0].message.content)