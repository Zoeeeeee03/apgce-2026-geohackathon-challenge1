# ============================================================
# STRATAMIND GEOLOGICAL RESOURCE LIBRARY
# ============================================================

RESOURCE_LIBRARY = {

    # --------------------------------------------------------
    # STRUCTURAL FEATURES
    # --------------------------------------------------------

    "normal_fault": {
        "feature": "Normal Fault",
        "category": "Structural",
        "keywords": [
            "normal fault", "extensional fault", "hanging wall",
            "footwall", "downthrown", "fault displacement"
        ],
        "visible_characteristics": [
            "Reflectors show vertical displacement across a discontinuity.",
            "The hanging wall is displaced downward relative to the footwall.",
            "Reflectors may terminate or become offset along a dipping fault plane."
        ],
        "interpretation": (
            "A possible normal fault associated with extensional deformation."
        ),
        "annotation_requirements": [
            "Trace the interpreted fault plane.",
            "Mark displaced reflectors.",
            "Label the hanging wall and footwall."
        ],
        "limitations": [
            "Verify using adjacent seismic lines and geological context.",
            "Poor imaging or fault shadows may obscure the fault plane."
        ],
        "sources": [
            "https://www.researchgate.net/figure/Seismic-profiles-illustrate-two-different-fault-geometries-from-the-Aphrodite-field-where_fig1_379080360"
        ]
    },

    "reverse_fault": {
        "feature": "Reverse Fault",
        "category": "Structural",
        "keywords": [
            "reverse fault", "thrust fault", "compressional fault",
            "hanging wall uplift", "footwall", "fault displacement"
        ],
        "visible_characteristics": [
            "Reflectors show vertical displacement across a dipping fault plane.",
            "The hanging wall is displaced upward relative to the footwall.",
            "Structural uplift and possible hanging-wall folding may occur."
        ],
        "interpretation": (
            "A possible reverse fault associated with crustal shortening "
            "and compressional deformation."
        ),
        "annotation_requirements": [
            "Trace the interpreted fault plane.",
            "Mark displaced reflectors and cutoff points.",
            "Label the hanging wall and footwall."
        ],
        "limitations": [
            "Verify fault geometry using regional seismic data and well logs.",
            "Velocity variations and fault shadows may affect imaging."
        ],
        "sources": [
            "https://academic.oup.com/gji/article/215/3/2134/5113496?login=false"
        ]
    },

    "anticline": {
        "feature": "Anticline",
        "category": "Structural",
        "keywords": [
            "anticline", "anticlinal fold", "structural closure",
            "crest", "fold", "arched reflectors"
        ],
        "visible_characteristics": [
            "Reflectors arch upward into a convex-upward geometry.",
            "Strata dip away from the fold crest on opposing limbs.",
            "A distinct structural crest may be visible."
        ],
        "interpretation": (
            "A possible anticline formed by folding, fault-related deformation, "
            "or movement of mobile substrates such as salt or shale."
        ),
        "annotation_requirements": [
            "Trace the fold axis or axial trace.",
            "Mark the crest and opposing limbs.",
            "Identify possible structural closure where supported."
        ],
        "limitations": [
            "A structural high alone does not prove a closed hydrocarbon trap.",
            "Velocity pull-up may create an apparent structural high.",
            "Older rocks in the core should not be assumed without age control."
        ],
        "sources": [
            "https://www.sciencedirect.com/topics/earth-and-planetary-sciences/anticlines",
            "https://www.researchgate.net/publication/373832657_Structural_Style_Analysis_of_Maqlob_Anticline_Using_Seismic_Data"
        ]
    },

    "syncline": {
        "feature": "Syncline",
        "category": "Structural",
        "keywords": [
            "syncline", "synclinal fold", "trough",
            "structural low", "concave upward"
        ],
        "visible_characteristics": [
            "Reflectors sag downward into a concave-upward geometry.",
            "Strata dip inward toward the fold axis.",
            "The structure forms a trough or structural low."
        ],
        "interpretation": (
            "A possible syncline formed by folding, subsidence, "
            "or differential compaction."
        ),
        "annotation_requirements": [
            "Trace the fold axis.",
            "Mark the structural low.",
            "Indicate the inward-dipping limbs."
        ],
        "limitations": [
            "A sagging reflection pattern may also represent differential compaction.",
            "Confirm fold geometry using adjacent seismic sections."
        ],
        "sources": [
            "https://geoexpro.com/finding-oil-in-synclines/",
            "https://dggs.alaska.gov/webpubs/dggs/ri/text/ri2011_003b.pdf"
        ]
    },

    "structural_high": {
        "feature": "Structural High",
        "category": "Structural / Trap Element",
        "keywords": [
            "structural high", "structural culmination",
            "horst", "uplift", "dome", "crest", "spill point"
        ],
        "visible_characteristics": [
            "Reflectors display local upward arching or positive relief.",
            "The crest is structurally higher than surrounding flanks.",
            "Possible bounding faults, thinning, truncation or onlap may occur."
        ],
        "interpretation": (
            "A possible positive structural feature caused by folding, "
            "fault-block uplift, inversion, or differential compaction."
        ),
        "annotation_requirements": [
            "Mark the structural crest.",
            "Trace associated faults or fold axes.",
            "Identify possible closing contours and spill points if mapping data exist.",
            "Indicate flank dip directions."
        ],
        "limitations": [
            "A 2D seismic section cannot establish complete 3D closure.",
            "Velocity pull-up may produce an apparent structural high.",
            "Depth conversion and velocity modelling may be required."
        ],
        "sources": [
            "http://homepage.ufp.pt/biblioteca/Seismic/Pages/Page7.htm",
            "https://www.researchgate.net/publication/336542200_Exploring_for_Structural_Traps"
        ]
    },

    "fault_closure": {
        "feature": "Fault Closure",
        "category": "Structural / Trap Element",
        "keywords": [
            "fault closure", "fault trap", "fault seal",
            "three way dip closure", "spill point", "fault bounded"
        ],
        "visible_characteristics": [
            "Reflectors terminate or are offset against a dipping fault plane.",
            "The fault may form a lateral boundary to a structural feature.",
            "Strata may dip away from the fault, creating possible three-way dip closure."
        ],
        "interpretation": (
            "A possible fault-dependent structural trap. Whether the fault "
            "acts as a seal or migration pathway requires additional evidence."
        ),
        "annotation_requirements": [
            "Trace the fault plane.",
            "Mark hanging-wall and footwall horizon cutoffs.",
            "Identify the structural crest and possible spill point where supported."
        ],
        "limitations": [
            "Seismic imaging alone cannot establish fault sealing capacity.",
            "Fault seal analysis and well data are needed.",
            "A single 2D section cannot prove areal closure."
        ],
        "sources": [
            "https://wiki.seg.org/wiki/Dictionary:Fault_seal_analysis",
            "https://www.researchgate.net/figure/Seismic-Interpretation-showing-some-faults-and-horizons-of-reservoirs-picked-on-seismic_fig3_331374082"
        ]
    },

    # --------------------------------------------------------
    # STRATIGRAPHIC FEATURES
    # --------------------------------------------------------

    "onlap": {
        "feature": "Onlap",
        "category": "Stratigraphic",
        "keywords": [
            "onlap", "reflection termination", "pinch out",
            "transgression", "basin margin"
        ],
        "visible_characteristics": [
            "Younger reflectors terminate progressively against an inclined older surface.",
            "Successive reflection terminations may shift landward or updip.",
            "The underlying surface may be an unconformity or depositional slope."
        ],
        "interpretation": (
            "A possible onlap pattern associated with transgression, "
            "increasing accommodation, or basin-margin deposition."
        ),
        "annotation_requirements": [
            "Trace the underlying bounding surface.",
            "Mark successive reflector termination points.",
            "Indicate the direction of onlap where known."
        ],
        "limitations": [
            "Thin onlapping beds may be below seismic resolution.",
            "Do not infer sea-level history from onlap alone."
        ],
        "sources": [
            "https://www.sepmstrata.org/page.aspx?pageid=229",
            "https://wiki.seg.org/wiki/Seismic_reflection_terminations"
        ]
    },

    "downlap": {
        "feature": "Downlap",
        "category": "Stratigraphic",
        "keywords": [
            "downlap", "clinoform", "progradation",
            "basinward", "reflection termination"
        ],
        "visible_characteristics": [
            "Inclined reflectors terminate downward against a lower surface.",
            "The lower surface is commonly horizontal or gently dipping.",
            "The reflection pattern may form a progradational wedge."
        ],
        "interpretation": (
            "A possible downlap pattern associated with progradation "
            "and basinward sediment accumulation."
        ),
        "annotation_requirements": [
            "Trace the basal downlap surface.",
            "Mark inclined reflector termination points.",
            "Indicate progradation direction where supported."
        ],
        "limitations": [
            "Low-amplitude terminations may be difficult to distinguish from noise.",
            "Migration artifacts can affect apparent reflector geometry."
        ],
        "sources": [
            "https://wiki.seg.org/wiki/Seismic_stratigraphy"
        ]
    },

    "pinch_out": {
        "feature": "Pinch-out",
        "category": "Stratigraphic / Trap Element",
        "keywords": [
            "pinch out", "pinch-out", "thinning",
            "wedge", "stratigraphic trap", "lateral termination"
        ],
        "visible_characteristics": [
            "A seismic package becomes progressively thinner laterally.",
            "Bounding reflectors converge toward a termination point.",
            "The layer may disappear against a bounding surface or facies transition."
        ],
        "interpretation": (
            "A possible lateral stratigraphic termination caused by non-deposition, "
            "erosion, condensation, or a lateral facies change."
        ),
        "annotation_requirements": [
            "Trace the top and base of the thinning package.",
            "Mark the apparent pinch-out location.",
            "Indicate the thinning direction."
        ],
        "limitations": [
            "Seismic tuning can obscure the true termination point.",
            "A reflector disappearing does not necessarily prove a geological pinch-out.",
            "Well-log calibration is recommended."
        ],
        "sources": [
            "https://www.researchgate.net/figure/Seismic-random-line-running-from-S-to-N-through-the-prospect-targeted-by-Well-1-The_fig3_321015153",
            "https://wiki.seg.org/wiki/Fort_Worth_Oil_Basin"
        ]
    },

    "unconformity": {
        "feature": "Unconformity",
        "category": "Stratigraphic / Sequence Boundary",
        "keywords": [
            "unconformity", "sequence boundary", "angular discordance",
            "erosion", "hiatus", "onlap", "truncation"
        ],
        "visible_characteristics": [
            "Reflectors may terminate by truncation below a boundary.",
            "Younger reflectors may onlap the same boundary.",
            "Underlying and overlying strata may show different dip orientations."
        ],
        "interpretation": (
            "A possible stratigraphic hiatus associated with erosion, "
            "non-deposition, or tectonic deformation."
        ),
        "annotation_requirements": [
            "Trace the unconformity surface.",
            "Mark truncation and onlap relationships.",
            "Indicate angular discordance where visible."
        ],
        "limitations": [
            "A disconformity may be difficult to identify from seismic data alone.",
            "Well and biostratigraphic data may be needed to confirm the hiatus."
        ],
        "sources": [
            "https://homepage.ufp.pt/biblioteca/SeismicStratigraphy/Pages/Page23.htm",
            "https://www.sepmstrata.org/page.aspx?pageid=229"
        ]
    },

    "truncation": {
        "feature": "Truncation",
        "category": "Stratigraphic / Sequence Boundary",
        "keywords": [
            "truncation", "erosional truncation",
            "reflector cutoff", "erosion", "unconformity"
        ],
        "visible_characteristics": [
            "Underlying reflectors terminate abruptly against a bounding surface.",
            "Dipping strata may be cut by a flatter or differently dipping surface.",
            "The termination may occur beneath an unconformity."
        ],
        "interpretation": (
            "A possible erosional or structural truncation caused by erosion, "
            "uplift, tilting, or cross-cutting deformation."
        ),
        "annotation_requirements": [
            "Trace the truncation surface.",
            "Mark the tips of terminated reflectors.",
            "Show the angular relationship between the packages."
        ],
        "limitations": [
            "Poor resolution can make termination points uncertain.",
            "Differentiate erosional truncation from fault displacement."
        ],
        "sources": [
            "https://homepage.ufp.pt/biblioteca/SeismicStratigraphy/Pages/Page22.htm"
        ]
    },

    # --------------------------------------------------------
    # DEPOSITIONAL FEATURES AND SEISMIC FACIES
    # --------------------------------------------------------

    "channel": {
        "feature": "Channel",
        "category": "Depositional / Seismic Geomorphology",
        "keywords": [
            "channel", "channel fill", "submarine canyon",
            "erosional scour", "meander", "channel axis"
        ],
        "visible_characteristics": [
            "U-shaped or V-shaped erosional geometry may cut underlying reflectors.",
            "The channel fill may be chaotic, transparent, or internally layered.",
            "Channel margins may be visible as concave erosional boundaries."
        ],
        "interpretation": (
            "A possible fluvial, deltaic, or deep-water sedimentary channel."
        ),
        "annotation_requirements": [
            "Trace the channel base and margins.",
            "Mark the channel axis where visible.",
            "Differentiate channel fill from surrounding strata."
        ],
        "limitations": [
            "Small channels may be below seismic resolution.",
            "A channel interpretation should ideally be checked using seismic slices or attributes."
        ],
        "sources": [
            "http://homepage.ufp.pt/biblioteca/WEBTurdiDepSystems/Pages/Page8.htm"
        ]
    },

    "reef_buildup": {
        "feature": "Reef Buildup",
        "category": "Depositional / Carbonate Facies",
        "keywords": [
            "reef", "carbonate buildup", "bioherm",
            "mounded reflector", "pinnacle", "carbonate platform"
        ],
        "visible_characteristics": [
            "Mounded, dome-shaped, or pinnacle-like reflection geometry.",
            "Internal reflections may be chaotic or low continuity.",
            "Flank reflections may dip outward from the buildup.",
            "Underlying reflectors may show apparent velocity pull-up."
        ],
        "interpretation": (
            "A possible carbonate buildup or reef body formed by carbonate accumulation."
        ),
        "annotation_requirements": [
            "Outline the buildup boundaries.",
            "Mark the crest and flanks.",
            "Differentiate internal facies from surrounding layered reflections."
        ],
        "limitations": [
            "Mounded geometry alone does not prove a reef.",
            "Velocity pull-up may distort underlying structures.",
            "Well data and regional depositional context are important."
        ],
        "sources": [
            "http://homepage.ufp.pt/biblioteca/WEBTurdiDepSystems/Pages/Page8.htm",
            "https://www.aapg.org/news-and-media/explorer/the-petroleum-geochemistry-toolkit-for-petroleum-exploration-and-development-2/"
        ]
    },

    "chaotic_reflections": {
        "feature": "Chaotic Reflections",
        "category": "Seismic Facies",
        "keywords": [
            "chaotic reflections", "chaotic seismic facies",
            "disorganized reflections", "mass transport",
            "slump", "debris flow"
        ],
        "visible_characteristics": [
            "Reflectors lack clear lateral continuity.",
            "Reflection segments are disorganized and variable in amplitude.",
            "The package may have a distinct basal boundary."
        ],
        "interpretation": (
            "Possible mass-transport deposits, slumps, debris flows, "
            "tectonic disruption, or heterogeneous geological bodies."
        ),
        "annotation_requirements": [
            "Outline the external boundary of the chaotic package.",
            "Mark the basal boundary where visible.",
            "Separate chaotic facies from adjacent coherent reflectors."
        ],
        "limitations": [
            "Poor signal-to-noise ratio can create apparent chaotic reflections.",
            "Do not assign a specific depositional origin without supporting evidence."
        ],
        "sources": [
            "https://wiki.seg.org/wiki/Dictionary:Seismic_facies_analysis"
        ]
    },

    "discontinuous_reflectors": {
        "feature": "Discontinuous Reflectors",
        "category": "Seismic Facies / Structural",
        "keywords": [
            "discontinuous reflectors", "broken reflections",
            "reflection continuity", "fault zone",
            "fracture", "facies transition"
        ],
        "visible_characteristics": [
            "Reflection events are fragmented or interrupted.",
            "Reflectors may terminate abruptly or lose lateral continuity.",
            "Zones of discontinuity may coincide with faults or facies boundaries."
        ],
        "interpretation": (
            "Possible structural disruption, lithological heterogeneity, "
            "or lateral stratigraphic changes."
        ),
        "annotation_requirements": [
            "Highlight zones of interrupted reflection continuity.",
            "Mark associated faults or boundaries only where supported.",
            "Compare with surrounding coherent reflection patterns."
        ],
        "limitations": [
            "Noise and acquisition footprints may create false discontinuities.",
            "Discontinuous reflections alone do not prove faulting."
        ],
        "sources": [
            "https://wiki.seg.org/wiki/Seismic_interpretation",
            "https://www.linkedin.com/pulse/seismic-facies-analysis-skillup-training-eg-xfque"
        ]
    }
}


# ============================================================
# SEARCH THE GEOLOGICAL RESOURCE LIBRARY
# ============================================================

def search_resource_library(question, max_results=4):

    question = question.lower()
    question_words = set(question.replace("-", " ").split())

    results = []

    for resource_id, resource in RESOURCE_LIBRARY.items():

        score = 0

        feature_name = resource["feature"].lower()

        if feature_name in question:
            score += 10

        for keyword in resource["keywords"]:

            keyword = keyword.lower()

            if keyword in question:
                score += 5

            keyword_words = set(keyword.replace("-", " ").split())
            score += len(question_words.intersection(keyword_words))

        if score > 0:
            results.append((score, resource_id, resource))

    results.sort(key=lambda item: item[0], reverse=True)

    return results[:max_results]


# ============================================================
# FORMAT REFERENCES FOR GEMMA
# ============================================================

def format_resource_context(question, max_results=4):

    results = search_resource_library(
        question,
        max_results=max_results
    )

    if not results:
        return (
            "No directly matching feature was found in the "
            "geological resource library. Use general geological "
            "knowledge cautiously and state uncertainty."
        )

    context = []

    for score, resource_id, resource in results:

        section = f"""
FEATURE: {resource['feature']}
CATEGORY: {resource['category']}

VISIBLE CHARACTERISTICS:
"""

        for item in resource["visible_characteristics"]:
            section += f"- {item}\n"

        section += "\nINTERPRETATION:\n"
        section += resource["interpretation"] + "\n"

        section += "\nANNOTATION REQUIREMENTS:\n"

        for item in resource["annotation_requirements"]:
            section += f"- {item}\n"

        section += "\nLIMITATIONS:\n"

        for item in resource["limitations"]:
            section += f"- {item}\n"

        section += "\nREFERENCE SOURCES:\n"

        for source in resource["sources"]:
            section += f"- {source}\n"

        context.append(section)

    return "\n\n".join(context)


