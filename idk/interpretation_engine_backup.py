import numpy as np
import os
import io
import re
import json
import base64
import requests

from PIL import Image, ImageDraw, ImageFont
from resource_library import format_resource_context


# ============================================================
# 1. GEMMA CONFIGURATION
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

GEMMA_CHAT_URL = f"{GEMMA_URL.rstrip('/')}/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GEMMA_API_KEY}",
    "Content-Type": "application/json"
}


# ============================================================
# 2. INTERPRETATION PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are StrataMind, an expert AI assistant for petroleum geology
and seismic interpretation.

Your primary objective is to identify and differentiate potential
STRUCTURAL TRAPS and STRATIGRAPHIC TRAPS from 2D seismic data.

SCIENTIFIC RULES:

1. Interpret only features supported by visible seismic evidence.
2. Separate direct observations from geological interpretations.
3. Never invent faults, horizons, reflector terminations or coordinates.
4. A seismic anomaly alone does not prove hydrocarbons.
5. Do not classify every geological structure as a hydrocarbon trap.
6. A 2D section cannot establish complete 3D closure.
7. State uncertainty and alternative geological explanations.
8. Distinguish possible traps from confirmed traps.
9. Confidence is an AI assessment, not a geological probability.
10. If evidence is insufficient, return an uncertain classification.

STRUCTURAL TRAP CRITERIA:

Structural traps are associated with deformation of geological
strata or structural geometry.

Look for:
- Anticlines and folded reflectors.
- Fault-bounded closures.
- Normal and reverse faults.
- Fault displacement and reflector offsets.
- Horst and tilted fault blocks.
- Structural highs.
- Dipping strata forming possible closures.

STRATIGRAPHIC TRAP CRITERIA:

Stratigraphic traps are associated with depositional geometry,
lateral facies changes or stratigraphic termination.

Look for:
- Sandstone pinch-outs.
- Onlap and downlap.
- Unconformity truncation.
- Channel-fill deposits.
- Reef or carbonate buildups.
- Lens-shaped sedimentary bodies.
- Changes in reflector continuity.
- Stratigraphic termination against sealing units.

COMBINATION TRAPS:

When structural deformation and stratigraphic termination
both contribute to potential trapping geometry, classify
the candidate as a combination trap.

IMPORTANT:
Do not classify a fault alone as a fault trap.
A potential trapping geometry and closure must be supported.
"""


INTERPRETATION_PROMPT = """
Analyse the supplied 2D seismic image as a petroleum geoscientist.

YOUR PRIMARY TASK:
Identify possible hydrocarbon trap geometries and distinguish
structural traps from stratigraphic traps.

PART 1: STRUCTURAL FEATURES

Investigate:
- Normal faults
- Reverse faults
- Fault displacement
- Anticlines
- Synclines
- Structural highs
- Horsts
- Tilted fault blocks
- Possible fault closures

PART 2: STRATIGRAPHIC FEATURES

Investigate:
- Onlap
- Downlap
- Pinch-outs
- Unconformities
- Truncation
- Channels
- Channel-fill deposits
- Reef buildups
- Discontinuous reflectors
- Stratigraphic termination

PART 3: SEISMIC INDICATORS

Investigate:
- Bright spots
- Dim spots
- Flat spots
- Amplitude anomalies
- Chaotic reflections
- Reflection continuity
- Reflection configuration
- Possible fluid-related indicators

PART 4: TRAP CLASSIFICATION

For every potential trap candidate, determine:

A. STRUCTURAL
B. STRATIGRAPHIC
C. COMBINATION
D. NOT ESTABLISHED

Do not force a trap classification when evidence is insufficient.

PART 5: GEOLOGICAL ASSESSMENT

Explain:
- Directly observed seismic evidence.
- Geological interpretation.
- Possible trapping mechanism.
- Potential reservoir geometry.
- Possible sealing mechanism.
- Alternative geological explanations.
- Additional evidence needed for validation.

PART 6: ANNOTATION

Return coordinates normalised between 0 and 1.

Coordinate system:
- x = horizontal image position, left to right.
- y = vertical image position, top to bottom.
- Coordinates must refer to the supplied image boundaries.

Use:
- line for faults and interpreted reflector trends.
- polygon for possible trap boundaries.
- box for regions of interest.
- point for individual anomalies.

Only annotate features that can be located confidently.

RETURN ONLY VALID JSON:

{
  "summary": "Overall seismic interpretation",
  "trap_assessment": {
    "classification": "structural",
    "trap_candidates": [
      {
        "candidate_id": "ST-001",
        "trap_type": "fault_bounded_closure",
        "classification": "structural",
        "confidence": 0.75,
        "observations": [
          "Visible reflector displacement",
          "Possible closure against fault plane"
        ],
        "evidence": "Explain the actual seismic evidence",
        "interpretation": "Possible geological interpretation",
        "trapping_mechanism": "Possible fault-dependent closure",
        "reservoir_geometry": "Not established",
        "seal_mechanism": "Not established",
        "alternative_explanations": [
          "Possible non-trapping fault geometry"
        ],
        "validation_required": [
          "Check adjacent seismic lines",
          "Evaluate fault seal"
        ],
        "shape": "polygon",
        "points": [
          [0.30, 0.30],
          [0.60, 0.30],
          [0.60, 0.65],
          [0.30, 0.65]
        ]
      }
    ]
  },
  "features": [
    {
      "type": "normal_fault",
      "category": "structural",
      "confidence": 0.80,
      "shape": "line",
      "points": [
        [0.30, 0.20],
        [0.40, 0.50],
        [0.48, 0.80]
      ],
      "evidence": "Observed reflector displacement",
      "interpretation": "Possible normal fault"
    }
  ]
}

PART 7: FEATURE CLASSIFICATION RULES

For every geological feature:

- Assign exactly one primary category.
- Use structural for faults, folds, structural highs and related deformation.
- Use stratigraphic for onlap, downlap, pinch-outs, unconformities and truncation.
- Use depositional for channels, channel fills and depositional geometries.
- Use amplitude for bright spots, dim spots, flat spots and amplitude anomalies.
- Do not count amplitude indicators as structural or stratigraphic features.

Each feature must contain:

- type
- category
- confidence (0 to 1)
- shape
- points
- observations
- evidence
- interpretation
- alternative_explanations

Confidence represents AI assessment confidence, not probability
of hydrocarbon presence.

Do not invent features or coordinates.
If evidence is unclear, omit the feature.


ADDITIONAL RULES:

- The example values above are illustrative, not actual findings.
- Replace all example content with observations from the supplied image.
- Use an empty trap_candidates list if no potential traps are identified.
- Use an empty features list if no reliable features are identified.
- Do not invent coordinates.
- Every candidate must have a classification and geological evidence.
- Confidence must be between 0 and 1.
- Do not describe a possible trap as confirmed.
"""



# ============================================================
# 3. GEMMA REQUEST
# ============================================================

def call_gemma_vision(
    seismic_png,
    user_query,
    geological_context="",
    temperature=0.1,
    max_tokens=2500
):
    """
    Send a seismic PNG and geological context to Gemma.
    """

    encoded_image = base64.b64encode(
        seismic_png
    ).decode("utf-8")

    image_url = f"data:image/png;base64,{encoded_image}"

    user_prompt = f"""
{INTERPRETATION_PROMPT}

GEOLOGICAL REFERENCE CONTEXT:
{geological_context}

USER REQUEST:
{user_query}

Analyse the attached seismic section.
Return the requested JSON only.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    payload = {
        "model": GEMMA_MODEL,
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
# 4. EXTRACT JSON
# ============================================================


def extract_json_response(response_text):
    """Extract structured JSON from Gemma's response."""

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        response_text,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    start = cleaned.find("{")

    if start == -1:
        raise ValueError("Gemma did not return JSON.")

    # Extract the first complete JSON object.
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(cleaned[start:])

    if not isinstance(result, dict):
        raise ValueError("Gemma response must be a JSON object.")

    return result


def validate_interpretation(interpretation):
    """
    Validate Gemma's geological interpretation.

    Keeps structural, stratigraphic and combination classifications
    separate and rejects invalid annotation coordinates.
    """

    if not isinstance(interpretation, dict):
        interpretation = {}

    valid_categories = {
        "structural",
        "stratigraphic",
        "amplitude",
        "depositional",
        "other"
    }

    valid_shapes = {
        "line",
        "polygon",
        "box",
        "point"
    }

    valid_classifications = {
        "structural",
        "stratigraphic",
        "combination",
        "not_established"
    }

    def clean_points(points):
        """Validate normalised image coordinates."""

        if not isinstance(points, list):
            return []

        valid = []

        for point in points:
            try:
                if not isinstance(point, (list, tuple)):
                    continue

                if len(point) != 2:
                    continue

                x = float(point[0])
                y = float(point[1])

                if (
                    not np.isfinite(x)
                    or not np.isfinite(y)
                    or not (0 <= x <= 1)
                    or not (0 <= y <= 1)
                ):
                    continue

                valid.append([x, y])

            except (TypeError, ValueError):
                continue

        return valid

    # ========================================================
    # 1. VALIDATE GENERAL FEATURES
    # ========================================================

    clean_features = []

    for feature in interpretation.get("features", []):

        if not isinstance(feature, dict):
            continue

        category = str(
            feature.get("category", "other")
        ).lower()

        if category not in valid_categories:
            category = "other"

        shape = str(
            feature.get("shape", "line")
        ).lower()

        if shape not in valid_shapes:
            continue

        points = clean_points(
            feature.get("points", [])
        )

        if not points:
            continue

        if shape == "line" and len(points) < 2:
            continue

        if shape == "polygon" and len(points) < 3:
            continue

        if shape == "box" and len(points) < 2:
            continue

        try:
            confidence = float(
                feature.get("confidence", 0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = float(
            np.clip(confidence, 0, 1)
        )

        
clean_features.append({
    "type": str(
        feature.get("type", "unknown")
    ),
    "category": category,
    "confidence": confidence,
    "confidence_label": "AI assessment confidence",
    "shape": shape,
    "points": points,
    "evidence": str(
        feature.get("evidence", "")
    ),
    "interpretation": str(
        feature.get("interpretation", "")
    ),
    "observations": (
        feature.get("observations", [])
        if isinstance(feature.get("observations", []), list)
        else []
    ),
    "alternative_explanations": (
        feature.get("alternative_explanations", [])
        if isinstance(feature.get("alternative_explanations", []), list)
        else []
    )
})



    # ========================================================
    # 2. VALIDATE TRAP ASSESSMENT
    # ========================================================

    trap_assessment = interpretation.get(
        "trap_assessment", {}
    )

    if not isinstance(trap_assessment, dict):
        trap_assessment = {}

    classification = str(
        trap_assessment.get(
            "classification",
            "not_established"
        )
    ).lower()

    if classification not in valid_classifications:
        classification = "not_established"

    clean_candidates = []

    candidates = trap_assessment.get(
        "trap_candidates", []
    )

    if not isinstance(candidates, list):
        candidates = []

    for index, candidate in enumerate(candidates, start=1):

        if not isinstance(candidate, dict):
            continue

        candidate_class = str(
            candidate.get("classification", "not_established")
        ).lower()

        if candidate_class not in valid_classifications:
            candidate_class = "not_established"

        shape = str(
            candidate.get("shape", "polygon")
        ).lower()

        if shape not in valid_shapes:
            continue

        points = clean_points(
            candidate.get("points", [])
        )

        if shape == "line" and len(points) < 2:
            continue

        if shape == "polygon" and len(points) < 3:
            continue

        if shape == "box" and len(points) < 2:
            continue

        if shape == "point" and len(points) < 1:
            continue

        try:
            confidence = float(
                candidate.get("confidence", 0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = float(
            np.clip(confidence, 0, 1)
        )

        def clean_list(value):
            if not isinstance(value, list):
                return []

            return [
                str(item)
                for item in value
                if item is not None
            ]

        clean_candidates.append({
            "candidate_id": str(
                candidate.get(
                    "candidate_id",
                    f"TC-{index:03d}"
                )
            ),
            "trap_type": str(
                candidate.get("trap_type", "unknown")
            ),
            "classification": candidate_class,
            "confidence": confidence,
            "observations": clean_list(
                candidate.get("observations", [])
            ),
            "evidence": str(
                candidate.get("evidence", "")
            ),
            "interpretation": str(
                candidate.get("interpretation", "")
            ),
            "trapping_mechanism": str(
                candidate.get("trapping_mechanism", "")
            ),
            "reservoir_geometry": str(
                candidate.get("reservoir_geometry", "")
            ),
            "seal_mechanism": str(
                candidate.get("seal_mechanism", "")
            ),
            "alternative_explanations": clean_list(
                candidate.get("alternative_explanations", [])
            ),
            "validation_required": clean_list(
                candidate.get("validation_required", [])
            ),
            "shape": shape,
            "points": points
        })

    return {
        "summary": str(
            interpretation.get(
                "summary",
                "No interpretation summary available."
            )
        ),
        "trap_assessment": {
            "classification": classification,
            "trap_candidates": clean_candidates
        },
        "features": clean_features
    }

# ============================================================
# 5. FEATURE SUMMARY
# ============================================================

def build_feature_summary(features):
    """
    Count geological features by category.
    Amplitude indicators are counted separately.
    """

    summary = {
        "structural": [],
        "stratigraphic": [],
        "amplitude": [],
        "depositional": [],
        "other": []
    }

    for feature in features:

        category = feature.get("category", "other")

        if category not in summary:
            category = "other"

        summary[category].append({
            "type": feature.get("type", "unknown"),
            "confidence": feature.get("confidence", 0),
            "evidence": feature.get("evidence", ""),
            "interpretation": feature.get("interpretation", ""),
            "shape": feature.get("shape", "point"),
            "points": feature.get("points", [])
        })

    return {
        "counts": {
            category: len(items)
            for category, items in summary.items()
        },
        "features": summary
    }



# ============================================================
# 6. RUN SEISMIC INTERPRETATION
# ============================================================

def identify_seismic_features(
    seismic_png,
    user_query=INTERPRETATION_PROMPT,
    geological_context=""
):
    """
    Run Gemma interpretation using an existing seismic PNG.

    This function does not read or reload any SEG-Y file.
    """

    if not seismic_png:
        raise ValueError("No seismic image was supplied.")

    resource_context = format_resource_context(
        user_query,
        max_results=8
    )

    combined_context = f"""
    GEOLOGICAL RESOURCE LIBRARY:
    {resource_context}

    ADDITIONAL GEOLOGICAL DATA:
    {geological_context}
    """

    response_text = call_gemma_vision(
        seismic_png=seismic_png,
        user_query=user_query,
        geological_context=combined_context
    )

    result = extract_json_response(response_text)

    return validate_interpretation(result)

# ============================================================
# 7. ANNOTATION COLOURS
# ============================================================

ANNOTATION_COLOURS = {
    "structural": (255, 65, 65),
    "stratigraphic": (30, 210, 100),
    "amplitude": (255, 210, 0),
    "other": (50, 180, 255)
}


# ============================================================
# 8. ANNOTATE SEISMIC IMAGE
# ============================================================



def annotate_seismic_image(seismic_png, interpretation):
    """
    Annotate seismic features and trap candidates identified by Gemma.

    Coordinates from the model are normalised (0-1) relative to the
    supplied seismic image.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io

    image = Image.open(io.BytesIO(seismic_png)).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    colours = {
        "structural": (255, 70, 70),
        "stratigraphic": (0, 220, 100),
        "amplitude": (255, 210, 0),
        "depositional": (0, 180, 220),
        "combination": (190, 90, 255),
        "not_established": (160, 160, 160),
        "other": (80, 160, 255)
    }

    annotated_features = []
    annotated_candidates = []

    def get_colour(category):
        return colours.get(str(category).lower(), colours["other"])

    def convert_points(points):
        converted = []

        for point in points or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            try:
                x = int(float(point[0]) * width)
                y = int(float(point[1]) * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                converted.append((x, y))
            except (TypeError, ValueError):
                continue

        return converted

    def draw_geometry(item, colour, label):
        points = convert_points(item.get("points", []))
        shape = str(item.get("shape", "point")).lower()

        if not points:
            return

        if shape == "polygon" and len(points) >= 3:
            draw.polygon(points, outline=colour, width=4)

        elif shape in ("line", "polyline") and len(points) >= 2:
            draw.line(points, fill=colour, width=4)

        elif shape in ("box", "rectangle") and len(points) >= 2:
            draw.rectangle(
                [points[0], points[1]],
                outline=colour,
                width=4
            )

        else:
            for x, y in points:
                radius = 7
                draw.ellipse(
                    (x-radius, y-radius, x+radius, y+radius),
                    fill=colour,
                    outline="white",
                    width=2
                )

        x, y = points[0]

        try:
            confidence = float(item.get("confidence", 0))
            confidence_text = f"{confidence:.0%}"
        except (TypeError, ValueError):
            confidence_text = "N/A"

        text = f"{label} ({confidence_text})"

        text_x = max(0, min(x, width - 150))
        text_y = max(0, min(y - 22, height - 20))

        draw.text(
            (text_x, text_y),
            text,
            fill=colour,
            stroke_width=2,
            stroke_fill="black"
        )

    # --------------------------------------------------------
    # ANNOTATE INDIVIDUAL GEOLOGICAL FEATURES
    # --------------------------------------------------------

    for index, feature in enumerate(
        interpretation.get("features", [])
    ):
        category = feature.get("category", "other")
        colour = get_colour(category)

        label = feature.get(
            "type",
            f"Feature {index + 1}"
        )

        draw_geometry(feature, colour, label)

        annotated_features.append(feature)

    # --------------------------------------------------------
    # ANNOTATE TRAP CANDIDATES
    # --------------------------------------------------------

    trap_assessment = interpretation.get(
        "trap_assessment", {}
    )

    for index, candidate in enumerate(
        trap_assessment.get("trap_candidates", [])
    ):
        classification = candidate.get(
            "classification",
            "not_established"
        )

        colour = get_colour(classification)

        label = candidate.get(
            "candidate_id",
            f"Trap {index + 1}"
        )

        draw_geometry(candidate, colour, label)

        annotated_candidates.append(candidate)

    # --------------------------------------------------------
    # ADD LEGEND
    # --------------------------------------------------------

    legend_items = [
        ("Structural", colours["structural"]),
        ("Stratigraphic", colours["stratigraphic"]),
        ("Amplitude", colours["amplitude"]),
        ("Combination", colours["combination"]),
        ("Other", colours["other"])
    ]

    legend_y = 10

    for label, colour in legend_items:
        draw.rectangle(
            (10, legend_y, 25, legend_y + 15),
            fill=colour
        )

        draw.text(
            (32, legend_y),
            label,
            fill="white",
            stroke_width=1,
            stroke_fill="black"
        )

        legend_y += 22

    # --------------------------------------------------------
    # EXPORT ANNOTATED IMAGE
    # --------------------------------------------------------

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")

    return (
        output_buffer.getvalue(),
        annotated_features,
        annotated_candidates
    )

# ============================================================
# 9. COMPLETE IMAGE INTERPRETATION
# ============================================================


def analyse_seismic_image(
    seismic_png,
    user_query=INTERPRETATION_PROMPT,
    geological_context=""
):
    """
    Complete seismic interpretation workflow.

    Gemma identifies geological features and trap candidates.
    Python validates, counts and annotates the interpretation.
    """

    interpretation = identify_seismic_features(
        seismic_png,
        user_query=user_query,
        geological_context=geological_context
    )

    interpretation = validate_interpretation(
        interpretation
    )

    # Count features by geological category
    feature_summary = build_feature_summary(
        interpretation["features"]
    )

    # Generate annotated seismic image
    annotated_png, features, candidates = annotate_seismic_image(
        seismic_png,
        interpretation
    )

    return {
        "original_image": seismic_png,
        "annotated_image": annotated_png,
        "features": features,
        "feature_summary": feature_summary,
        "trap_candidates": candidates,
        "trap_assessment": interpretation["trap_assessment"],
        "summary": interpretation["summary"],
        "interpretation": interpretation
    }



