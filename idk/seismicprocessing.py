import streamlit as st
import io
import os
import tempfile

import numpy as np
import pandas as pd
import segyio
import plotly.graph_objects as go


# ============================================================
# 0. READ CHECKSHOT DATA
# ============================================================

def read_checkshot_txt(file_bytes, filename="checkshot.txt"):
    """
    Read checkshot TXT data containing:
    TVDSS (m) and TWT (ms).

    Lines beginning with # are treated as comments.
    """

    if not file_bytes:
        raise ValueError("The checkshot file is empty.")

    text = file_bytes.decode("utf-8", errors="replace")

    # Extract well name from comment headers
    well_name = None

    for line in text.splitlines():

        line = line.strip()

        if line.lower().startswith("#p-"):
            well_name = line.lstrip("#").strip()

    # Read numerical data, ignoring comment lines
    try:
        data = pd.read_csv(
            io.StringIO(text),
            sep=r"[\s,]+",
            comment="#",
            header=None,
            usecols=[0, 1],
            engine="python"
        )

    except Exception as error:
        raise ValueError(
            f"Unable to read checkshot file: {error}"
        )

    data.columns = ["TVDSS_m", "TWT_ms"]

    # Convert values to numerical format
    data["TVDSS_m"] = pd.to_numeric(
        data["TVDSS_m"], errors="coerce"
    )

    data["TWT_ms"] = pd.to_numeric(
        data["TWT_ms"], errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(data) < 2:
        raise ValueError(
            "At least two valid checkshot measurements are required."
        )

    # Sort by depth
    data = data.sort_values("TVDSS_m")

    data = data.drop_duplicates(
        subset="TVDSS_m"
    ).reset_index(drop=True)

    # Check travel-time progression
    if np.any(np.diff(data["TWT_ms"]) < 0):
        raise ValueError(
            "Checkshot TWT decreases with increasing TVDSS. "
            "Please inspect the source data."
        )

    return {
        "filename": filename,
        "well_name": well_name,
        "data": data,
        "number_of_points": len(data),
        "depth_reference": "TVDSS",
        "time_reference": "TWT"
    }

# ============================================================
# 1. READ SEG-Y FILE
# ============================================================

def read_segy(file_bytes, filename="seismic.sgy"):
    """
    Read a SEG-Y file and extract its seismic amplitude data
    and basic metadata.

    Parameters
    ----------
    file_bytes : bytes
        Uploaded SEG-Y file content.
    filename : str
        Original SEG-Y filename.

    Returns
    -------
    dict
        Seismic amplitude matrix and metadata.
    """

    if not file_bytes:
        raise ValueError("The uploaded SEG-Y file is empty.")

    temp_path = None

    try:
        # Save uploaded bytes temporarily because segyio
        # reads SEG-Y files from a file path.

        suffix = os.path.splitext(filename)[1] or ".sgy"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Read SEG-Y file

        with segyio.open(
            temp_path,
            "r",
            ignore_geometry=True
        ) as segy_file:

            # Read seismic traces
            amplitudes = np.asarray(
                segy_file.trace.raw[:],
                dtype=np.float32
            )

            # Basic metadata
            trace_count = amplitudes.shape[0]
            sample_count = amplitudes.shape[1]

            sample_interval_us = int(
                segy_file.bin[segyio.BinField.Interval]
            )

            sample_interval_ms = (
                sample_interval_us / 1000
                if sample_interval_us > 0
                else None
            )

            # Sample axis in milliseconds
            if sample_interval_ms is not None:
                sample_axis = (
                    np.arange(sample_count)
                    * sample_interval_ms
                )
            else:
                sample_axis = np.arange(sample_count)

            # Basic amplitude statistics
            valid_amplitudes = amplitudes[
                np.isfinite(amplitudes)
            ]

            if valid_amplitudes.size > 0:

                amplitude_stats = {
                    "minimum": float(
                        np.min(valid_amplitudes)
                    ),
                    "maximum": float(
                        np.max(valid_amplitudes)
                    ),
                    "mean": float(
                        np.mean(valid_amplitudes)
                    ),
                    "standard_deviation": float(
                        np.std(valid_amplitudes)
                    ),
                    "rms": float(
                        np.sqrt(
                            np.mean(
                                valid_amplitudes ** 2
                            )
                        )
                    )
                }

            else:
                amplitude_stats = {}

            # Extract selected trace header information
            header_fields = {}

            selected_fields = {
                "CDP": segyio.TraceField.CDP,
                "OFFSET": segyio.TraceField.offset,
                "TRACE_SEQUENCE_LINE":
                    segyio.TraceField.TRACE_SEQUENCE_LINE
            }

            for field_name, field_code in selected_fields.items():

                try:
                    values = np.asarray(
                        segy_file.attributes(field_code)[:]
                    )

                    header_fields[field_name] = values

                except Exception:
                    header_fields[field_name] = None

            # Output
            result = {
                "filename": filename,
                "amplitudes": amplitudes,
                "trace_count": trace_count,
                "sample_count": sample_count,
                "sample_interval_us": sample_interval_us,
                "sample_interval_ms": sample_interval_ms,
                "sample_axis": sample_axis,
                "amplitude_stats": amplitude_stats,
                "trace_headers": header_fields
            }

        return result

    except Exception as error:

        raise RuntimeError(
            f"Unable to read SEG-Y file '{filename}': {error}"
        ) from error

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 2. GENERATE 2D SEISMIC DISPLAY
# ============================================================

def generate_seismic_figure(
    seismic_data,
    clip_percentile=99.5
):
    """
    Generate a 2D seismic section using Plotly.

    X-axis: Trace number
    Y-axis: Two-way travel time (ms), when available
    Colour: Seismic amplitude
    """

    amplitudes = seismic_data["amplitudes"]
    sample_axis = seismic_data["sample_axis"]

    # Calculate display clipping to improve visibility
    valid = amplitudes[np.isfinite(amplitudes)]

    if valid.size == 0:
        raise ValueError("No valid seismic amplitudes found.")

    clip_value = np.percentile(
        np.abs(valid),
        clip_percentile
    )

    if clip_value <= 0:
        clip_value = 1.0

    trace_axis = np.arange(
        amplitudes.shape[0]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=amplitudes.T,
            x=trace_axis,
            y=sample_axis,
            colorscale="RdBu",
            zmin=-clip_value,
            zmax=clip_value,
            zmid=0,
            colorbar=dict(
                title="Amplitude"
            ),
            hovertemplate=(
                "Trace: %{x}<br>"
                "Time: %{y:.2f}<br>"
                "Amplitude: %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=seismic_data["filename"],
        xaxis_title="Trace Number",
        yaxis_title=(
            "Two-Way Travel Time (ms)"
            if seismic_data["sample_interval_ms"]
            else "Sample Number"
        ),
        height=650,
        template="plotly_white",
        margin=dict(
            l=60,
            r=30,
            t=60,
            b=50
        )
    )

    # Two-way travel time increases downward
    fig.update_yaxes(
        autorange="reversed"
    )

    return fig


# ============================================================
# 3. GENERATE SEISMIC SUMMARY
# ============================================================

def get_seismic_summary(seismic_data):

    stats = seismic_data["amplitude_stats"]

    summary = {
        "Filename": seismic_data["filename"],
        "Number of traces": seismic_data["trace_count"],
        "Samples per trace": seismic_data["sample_count"],
        "Sampling interval (us)":
            seismic_data["sample_interval_us"],
        "Sampling interval (ms)":
            seismic_data["sample_interval_ms"],
        "Minimum amplitude": stats.get("minimum"),
        "Maximum amplitude": stats.get("maximum"),
        "Mean amplitude": stats.get("mean"),
        "Amplitude standard deviation":
            stats.get("standard_deviation"),
        "RMS amplitude": stats.get("rms")
    }

    return pd.DataFrame(
        summary.items(),
        columns=["Parameter", "Value"]
    )


# ============================================================
# 4. MEMORY-EFFICIENT SEG-Y PREVIEW READER
# ============================================================

def read_segy_preview(
    filepath,
    max_traces=1200,
    sample_step=1
):
    """
    Read a manageable selection of traces from a SEG-Y file.

    This avoids loading the entire 5 GB dataset into RAM.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    with segyio.open(
        filepath,
        "r",
        ignore_geometry=True
    ) as f:

        trace_count = f.tracecount
        sample_count = len(f.samples)

        # Select traces evenly across the seismic section
        trace_step = max(
            1,
            int(np.ceil(trace_count / max_traces))
        )

        selected_traces = np.arange(
            0,
            trace_count,
            trace_step
        )

        # Read selected traces only
        amplitudes = np.asarray(
            f.trace.raw[
                0:trace_count:trace_step
            ],
            dtype=np.float32
        )

        # Reduce sample density if requested
        amplitudes = amplitudes[:, ::sample_step]

        sample_interval_us = int(
            f.bin[segyio.BinField.Interval]
        )

        sample_interval_ms = (
            sample_interval_us / 1000
        )

        sample_axis = (
            np.arange(amplitudes.shape[1])
            * sample_interval_ms
            * sample_step
        )

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "amplitudes": amplitudes,
        "trace_axis": selected_traces,
        "sample_axis": sample_axis,
        "trace_count": trace_count,
        "sample_count": sample_count,
        "displayed_traces": len(selected_traces),
        "sample_interval_us": sample_interval_us,
        "sample_interval_ms": sample_interval_ms
    }


# ============================================================
# 5. INTERACTIVE SEISMIC DISPLAY
# ============================================================

def plot_seismic_section(
    seismic_data,
    clip_percentile=99.5,
    colorscale="RdBu"
):

    amplitudes = seismic_data["amplitudes"]

    valid = amplitudes[np.isfinite(amplitudes)]

    if valid.size == 0:
        raise ValueError("No valid amplitude data.")

    clip_value = np.percentile(
        np.abs(valid),
        clip_percentile
    )

    if clip_value <= 0:
        clip_value = 1.0

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=amplitudes.T,
            x=seismic_data["trace_axis"],
            y=seismic_data["sample_axis"],
            colorscale=colorscale,
            zmin=-clip_value,
            zmax=clip_value,
            zmid=0,
            colorbar=dict(title="Amplitude"),
            hovertemplate=(
                "Trace: %{x}<br>"
                "Time: %{y:.2f} ms<br>"
                "Amplitude: %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=seismic_data["filename"],
        xaxis_title="Trace Number",
        yaxis_title="Two-Way Travel Time (ms)",
        height=750,
        template="plotly_white",
        dragmode="zoom"
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    return fig

    
# ============================================================
# 6. WELL-TO-SEISMIC CORRELATION OVERLAY
# ============================================================

def add_well_correlation_overlay(
    fig,
    seismic_data,
    correlation_data
):
    """
    Add a user-selected well trace and optional checkshot
    calibration points to the seismic section.

    The well trace is user-selected and is not an
    automatically verified well location.
    """

    if not correlation_data:
        return fig

    selected_trace = correlation_data["trace"]

    sample_axis = np.asarray(
        seismic_data["sample_axis"],
        dtype=float
    )

    if len(sample_axis) == 0:
        return fig

    time_min = float(np.nanmin(sample_axis))
    time_max = float(np.nanmax(sample_axis))

    # --------------------------------------------------------
    # 1. USER-SELECTED WELL TRACE
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                selected_trace,
                selected_trace
            ],

            y=[
                time_min,
                time_max
            ],

            mode="lines",

            line=dict(
                color="#FFD700",
                width=2.5,
                dash="dash"
            ),

            name="User-selected well trace",

            hovertemplate=(
                "Selected trace: %{x}"
                "<extra>Well trace</extra>"
            ),

            showlegend=True
        )
    )

    # --------------------------------------------------------
    # 2. OPTIONAL CHECKSHOT CALIBRATION POINTS
    # --------------------------------------------------------

    checkshot_df = correlation_data.get(
        "checkshot_data"
    )

    if checkshot_df is not None and not checkshot_df.empty:

        checkshot_df = checkshot_df.copy()

        checkshot_df["TWT_ms"] = pd.to_numeric(
            checkshot_df["TWT_ms"],
            errors="coerce"
        )

        checkshot_df["TVDSS_m"] = pd.to_numeric(
            checkshot_df["TVDSS_m"],
            errors="coerce"
        )

        checkshot_df = checkshot_df.dropna(
            subset=["TWT_ms", "TVDSS_m"]
        )

        visible_points = checkshot_df[
            (checkshot_df["TWT_ms"] >= time_min) &
            (checkshot_df["TWT_ms"] <= time_max)
        ]

        if not visible_points.empty:

            fig.add_trace(
                go.Scatter(
                    x=[
                        selected_trace
                    ] * len(visible_points),

                    y=visible_points["TWT_ms"],

                    mode="markers",

                    marker=dict(
                        color="#00A6A6",
                        size=10,
                        symbol="diamond",
                        line=dict(
                            color="white",
                            width=1
                        )
                    ),

                    name="Checkshot calibration points",

                    customdata=visible_points[
                        ["TVDSS_m"]
                    ].to_numpy(),

                    hovertemplate=(
                        "Trace: %{x}<br>"
                        "TWT: %{y:.2f} ms<br>"
                        "TVDSS: %{customdata[0]:.2f} m"
                        "<extra>Checkshot calibration</extra>"
                    ),

                    showlegend=True
                )
            )

        else:

            st.warning(
                "The checkshot TWT values do not overlap "
                "the displayed seismic time range. "
                "No checkshot markers were drawn."
            )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="Correlation Annotations",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig
    # --------------------------------------------------------
    # 1. Draw selected well trace marker
    # --------------------------------------------------------

    fig.add_vline(
        x=selected_trace,
        line=dict(
            color="#FFD700",
            width=2,
            dash="dash"
        )
    )

    fig.add_annotation(
        x=selected_trace,
        y=time_min,
        text="Selected well trace",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#FFD700",
        font=dict(
            color="#303F32",
            size=11
        ),
        bgcolor="#FFF4B8",
        bordercolor="#D6B84C",
        borderwidth=1,
        yshift=-25
    )

    # --------------------------------------------------------
    # 2. Add optional checkshot calibration points
    # --------------------------------------------------------

    checkshot_df = correlation_data.get("checkshot_data")

    if checkshot_df is not None and not checkshot_df.empty:

        checkshot_df = checkshot_df.copy()

        checkshot_df["TWT_ms"] = pd.to_numeric(
            checkshot_df["TWT_ms"],
            errors="coerce"
        )

        checkshot_df["TVDSS_m"] = pd.to_numeric(
            checkshot_df["TVDSS_m"],
            errors="coerce"
        )

        checkshot_df = checkshot_df.dropna(
            subset=["TWT_ms", "TVDSS_m"]
        )

        # Only display checkshot points within seismic time range
        visible_points = checkshot_df[
            (checkshot_df["TWT_ms"] >= time_min) &
            (checkshot_df["TWT_ms"] <= time_max)
        ]

        if not visible_points.empty:

            fig.add_trace(
                go.Scatter(
                    x=[
                        selected_trace
                    ] * len(visible_points),

                    y=visible_points["TWT_ms"],

                    mode="markers+text",

                    marker=dict(
                        color="#00A6A6",
                        size=9,
                        symbol="diamond",
                        line=dict(
                            color="white",
                            width=1
                        )
                    ),

                    text=[
                        f"TVDSS {depth:.1f} m"
                        for depth in visible_points["TVDSS_m"]
                    ],

                    textposition="middle right",

                    textfont=dict(
                        color="#006666",
                        size=10
                    ),

                    name="Checkshot calibration",

                    customdata=visible_points[
                        ["TVDSS_m"]
                    ].to_numpy(),

                    hovertemplate=(
                        "Selected trace: %{x}<br>"
                        "TWT: %{y:.2f} ms<br>"
                        "TVDSS: %{customdata[0]:.2f} m"
                        "<extra>Checkshot calibration</extra>"
                    )
                )
            )

        else:

            st.warning(
                "The checkshot TWT values do not overlap "
                "the displayed seismic time range. "
                "No checkshot markers were drawn."
            )

    return fig


    
