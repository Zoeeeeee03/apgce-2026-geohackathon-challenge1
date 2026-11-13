
import os
import io
import re
import json
import base64
import tempfile
import requests

import numpy as np
import segyio
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. GEMMA / vLLM CONFIGURATION
# ============================================================

GEMMA_URL = os.getenv(
    "GEOH_VLLM_URL",
    "http://172.28.34.1:8000/v1"
)

MODEL_NAME = os.getenv(
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

HEADERS = {
    "Authorization": f"Bearer {GEMMA_API_KEY}",
    "Content-Type": "application/json"
}


# ============================================================
# 2. SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are StrataMind, a geoscience AI copilot for Team 04
in the APGCE GeoHackathon Challenge 1.

You interpret petroleum geology, reservoir traps,
wireline logs and seismic evidence.

STRICT RULES:

1. Give the primary answer first.
2. Distinguish observations from geological interpretations.
3. Never invent seismic features, depths, coordinates or well data.
4. A bright spot alone does not prove hydrocarbons.
5. A single 2D seismic section cannot establish a complete
   3D structural closure.
6. Explain uncertainty and alternative interpretations.
7. Keep answers concise and suitable for a technical demonstration.
"""


SEISMIC_ANALYSIS_PROMPT = """
You are StrataMind's seismic interpretation engine.

Analyse the supplied 2D seismic section.

Identify possible:

STRUCTURAL FEATURES:
- Normal faults
- Reverse faults
- Fault displacement
- Anticlines
- Synclines
- Structural highs
- Possible fault closures

STRATIGRAPHIC FEATURES:
- Onlap
- Downlap
- Unconformities
- Truncation
- Pinch-outs
- Channels
- Reef buildups

AMPLITUDE FEATURES:
- Bright spots
- Dim spots
- Flat spots
- Chaotic reflections
- Discontinuous reflectors

Return ONLY valid JSON using this structure:

{
  "features": [
    {
      "type": "normal_fault",
      "category": "structural",
      "confidence": 0.75,
      "shape": "line",
      "points": [
        [0.30, 0.20],
        [0.40, 0.50],
        [0.48, 0.80]
      ],
      "evidence": "Reflector displacement across a possible fault plane",
      "interpretation": "Possible normal fault"
    }
  ],
  "summary": "Brief geological interpretation"
}

COORDINATE RULES:

- Coordinates must be NORMALISED between 0 and 1.
- x = horizontal position from left to right.
- y = vertical position from top to bottom.
- Coordinates refer to the supplied image boundaries.
- Use shape=line for faults and interpreted horizons.
- Use shape=polygon for possible trap boundaries or regions.
- Use shape=box for rectangular regions.
- Use shape=point for individual locations.
- Include enough points to represent the feature.
- Do not invent coordinates for features that cannot be
  located confidently.
- Return an empty features list if no reliable features
  can be identified.

INTERPRETATION RULES:

- Confidence is a model-estimated score, not a verified
  geological probability.
- A possible trap must be distinguished from a confirmed trap.
- Do not claim hydrocarbon accumulation without supporting evidence.
- State when image quality or seismic evidence is insufficient.
"""


# ============================================================
# 3. GEOLOGICAL KNOWLEDGE BASE
# ============================================================

RAG_KNOWLEDGE_BASE = {

    "Well P-1": """
[WELL CONTEXT: WELL P-1]

- Trap System: Structural Trap
- Trap Architecture: Normal Fault Block / Faulted Dip Closure
- Seismic Inline Evidence:
  Inline 1420 displays approximately 45 ms TWT vertical
  reflector displacement along the main normal fault plane.
  Horizon A terminates sharply against the downthrown fault wall.

- Wireline Log Evidence:
  Low Vsh (<0.12), effective porosity 18.5%,
  hydrocarbon saturation 76%, deep resistivity 150 ohm-m
  across the 2150-2190 m interval.

- Geological Mechanism:
  Downthrown reservoir sandstone may be laterally sealed
  updip against impermeable shale across the fault plane.
""",

    "Well M-1": """
[WELL CONTEXT: WELL M-1]

- Trap System: Stratigraphic Trap
- Trap Architecture: Angular Unconformity /
  Channelized Sand Body Truncation

- Seismic Inline Evidence:
  Inline 1580 exhibits a high-amplitude reflector pinch-out
  beneath Unconformity U-1.
  Amplitude extraction indicates a sinuous channel geometry
  approximately 350 m wide.

- Wireline Log Evidence:
  Blocky Gamma Ray profile, Vsh <0.10,
  porosity 18.5% and Vp/Vs ratio 1.62
  at the 2340-2375 m interval.

- Geological Mechanism:
  Sub-unconformity truncation where reservoir-quality
  channel sands may be sealed by overlying marine mudstones.
"""
}


def get_well_context(selected_well):
    return RAG_KNOWLEDGE_BASE.get(
        selected_well,
        f"""
        [ACTIVE WELL: {selected_well}]
        No detailed well-specific geological context
        is currently available.
        Do not assume an established trap.
        """
    )


# ============================================================
# 4. GEMMA HTTP REQUEST
# ============================================================

def call_gemma(messages, temperature=0.1, max_tokens=1500):

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        GEMMA_CHAT_URL,
        headers=HEADERS,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ============================================================
# 5. EXISTING WELL COPILOT FUNCTION
# ============================================================

def get_copilot_response(selected_well, user_query):

    well_context = get_well_context(selected_well)

    user_prompt = f"""
ACTIVE WELL: {selected_well}

{well_context}

USER QUESTION:
{user_query}

Answer using only the supplied geological context.
Clearly identify missing information.
"""

    try:

        answer = call_gemma([
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ], max_tokens=500)

        return answer

    except Exception as error:

        return f"Gemma server error: {error}"


# ============================================================
# 6. READ 2D SEG-Y FILE
# ============================================================

def process_segy_2d(file_bytes, filename="seismic.sgy"):

    temp_path = None
    plt.close("all")

    try:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".sgy",
            delete=False
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Read SEG-Y traces
        with segyio.open(
            temp_path,
            "r",
            ignore_geometry=True
        ) as segy_file:

            seismic_data = np.asarray(
                segy_file.trace.raw[:],
                dtype=np.float32
            )

            interval_us = int(
                segy_file.bin[segyio.BinField.Interval]
            )

            sample_format = int(
                segy_file.bin[segyio.BinField.Format]
            )

            trace_count, sample_count = seismic_data.shape

            try:
                trace_samples = len(segy_file.samples)
            except Exception:
                trace_samples = sample_count

            if interval_us > 0:

                sample_interval_ms = interval_us / 1000.0

                vertical_values = (
                    np.arange(sample_count)
                    * sample_interval_ms
                )

                vertical_label = "Two-Way Travel Time (ms)"

            else:

                vertical_values = np.arange(sample_count)

                vertical_label = "Sample Index"

        # Check data
        seismic_data = np.nan_to_num(
            seismic_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if seismic_data.size == 0:
            raise ValueError("No seismic traces found.")

        # Amplitude clipping for display
        amplitude_limit = np.percentile(
            np.abs(seismic_data),
            98
        )

        if amplitude_limit <= 0:
            raise ValueError("Seismic amplitude values are zero.")

        # Generate seismic section
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.imshow(
            seismic_data.T,
            cmap="seismic",
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=[
                0,
                trace_count - 1,
                vertical_values[-1],
                vertical_values[0]
            ],
            vmin=-amplitude_limit,
            vmax=amplitude_limit
        )

        ax.set_title(
            f"2D Seismic Section: {filename}",
            fontsize=14
        )

        ax.set_xlabel("Trace Number")
        ax.set_ylabel(vertical_label)

        ax.grid(
            visible=True,
            alpha=0.2
        )

        fig.tight_layout()

        # Export seismic section as PNG
        image_buffer = io.BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        seismic_png = image_buffer.getvalue()

        metadata = {
            "filename": filename,
            "trace_count": trace_count,
            "sample_count": sample_count,
            "sample_interval_us": interval_us,
            "sample_format": sample_format,
            "vertical_label": vertical_label,
            "amplitude_min": float(np.min(seismic_data)),
            "amplitude_max": float(np.max(seismic_data)),
            "amplitude_limit": float(amplitude_limit)
        }

        return seismic_data, seismic_png, metadata

    except Exception as error:

        raise RuntimeError(
            f"SEG-Y processing failed: {error}"
        )

    finally:

        plt.close("all")

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 7. CONVERT IMAGE TO GEMMA VISION INPUT
# ============================================================

def image_to_data_url(image_bytes):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded_image
    )


# ============================================================
# 8. EXTRACT JSON FROM GEMMA RESPONSE
# ============================================================

def extract_json_response(response_text):

    # Remove Markdown code fences if returned
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        response_text,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Gemma did not return valid JSON.")

    return json.loads(cleaned[start:end + 1])


# ============================================================
# 9. GEMMA SEISMIC FEATURE DETECTION
# ============================================================

def identify_seismic_features(
    seismic_png,
    selected_well="All wells",
    user_query="Identify possible structural and stratigraphic traps."
):

    well_context = get_well_context(selected_well)

    image_url = image_to_data_url(seismic_png)

    user_content = [
        {
            "type": "text",
            "text": f"""
{SEISMIC_ANALYSIS_PROMPT}

ACTIVE WELL CONTEXT:
{well_context}

USER REQUEST:
{user_query}

Analyse the attached seismic section.
Return the requested JSON only.
"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    ]

    try:

        response_text = call_gemma(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0.1,
            max_tokens=2500
        )

        result = extract_json_response(response_text)

        if not isinstance(result.get("features"), list):
            raise ValueError("Invalid features list in Gemma response.")

        return result

    except Exception as error:

        return {
            "features": [],
            "summary": f"Seismic interpretation failed: {error}"
        }


# ============================================================
# 10. ANNOTATION ENGINE
# ============================================================

ANNOTATION_COLOURS = {
    "structural": (255, 65, 65),
    "stratigraphic": (30, 210, 100),
    "amplitude": (255, 210, 0),
    "other": (50, 180, 255)
}


def annotate_seismic_image(
    seismic_png,
    interpretation,
    line_width=4
):

    image = Image.open(
        io.BytesIO(seismic_png)
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            max(14, int(width * 0.015))
        )
    except Exception:
        font = ImageFont.load_default()

    features = interpretation.get(
        "features",
        []
    )

    annotated_features = []

    for feature in features:

        try:

            category = feature.get(
                "category",
                "other"
            ).lower()

            colour = ANNOTATION_COLOURS.get(
                category,
                ANNOTATION_COLOURS["other"]
            )

            feature_type = feature.get(
                "type",
                "unknown"
            )

            shape = feature.get(
                "shape",
                "line"
            ).lower()

            points = feature.get(
                "points",
                []
            )

            # Validate normalised coordinates
            valid_points = []

            for point in points:

                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                ):
                    continue

                x, y = float(point[0]), float(point[1])

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                ):
                    continue

                px = int(x * (width - 1))
                py = int(y * (height - 1))

                valid_points.append((px, py))

            if not valid_points:
                continue

            # Draw feature
            if shape == "line" and len(valid_points) >= 2:

                draw.line(
                    valid_points,
                    fill=colour,
                    width=line_width,
                    joint="curve"
                )

            elif shape == "polygon" and len(valid_points) >= 3:
      
                draw.polygon(
                    valid_points,
                    outline=colour,
                    width=line_width
    )       

            elif shape == "box" and len(valid_points) >= 2:

                draw.rectangle(
                    [
                        valid_points[0],
                        valid_points[1]
                    ],
                    outline=colour,
                    width=line_width
                )

            elif shape == "point":

                x, y = valid_points[0]

                radius = 7

                draw.ellipse(
                    [
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius
                    ],
                    fill=colour,
                    outline="white",
                    width=2
                )

            else:
                continue

            # Feature label
            confidence = feature.get(
                "confidence",
                None
            )

            if confidence is not None:
                label = (
                    f"{feature_type} "
                    f"({float(confidence) * 100:.0f}%)"
                )
            else:
                label = feature_type

            label_x, label_y = valid_points[0]

            text_box = draw.textbbox(
                (label_x, label_y),
                label,
                font=font
            )

            draw.rectangle(
                (
                    text_box[0] - 3,
                    text_box[1] - 3,
                    text_box[2] + 3,
                    text_box[3] + 3
                ),
                fill=(20, 20, 20)
            )

            draw.text(
                (label_x, label_y),
                label,
                fill=colour,
                font=font
            )

            annotated_features.append(feature)

        except Exception:
            # Skip invalid annotations instead of stopping
            continue

    # Save annotated image
    output_buffer = io.BytesIO()

    image.save(
        output_buffer,
        format="PNG"
    )

    return output_buffer.getvalue(), annotated_features


# ============================================================
# 11. COMPLETE SEG-Y INTERPRETATION WORKFLOW
# ============================================================

def analyse_segy_file(
    file_bytes,
    filename="seismic.sgy",
    selected_well="All wells",
    user_query="Identify possible structural and stratigraphic traps."
):

    # Step 1: Extract seismic data
    seismic_data, original_png, metadata = process_segy_2d(
        file_bytes,
        filename
    )

    # Step 2: Gemma analyses the seismic image
    interpretation = identify_seismic_features(
        original_png,
        selected_well,
        user_query
    )

    # Step 3: Python annotates the image
    annotated_png, features = annotate_seismic_image(
        original_png,
        interpretation
    )

    return {
        "seismic_data": seismic_data,
        "original_image": original_png,
        "annotated_image": annotated_png,
        "metadata": metadata,
        "features": features,
        "summary": interpretation.get(
            "summary",
            "No interpretation summary available."
        )
    }
