
from seismicprocessing import (
    read_segy_preview,
    plot_seismic_section
)

# Full Stack SEG-Y location
SEGY_PATH = (
    "/data/epic20/APGCE/input_data/"
    "challenge1_challenge2/stack/"
    "Challenge1_Challenge2_Full_Stack_cropped.sgy"
)

print("Reading Full Stack seismic data...")

# Read seismic data
seismic = read_segy_preview(
    SEGY_PATH,
    max_traces=1200,
    sample_step=1
)

# Display metadata
print("\n===== SEISMIC INFORMATION =====")

print("Filename:", seismic["filename"])
print("Total traces:", seismic["trace_count"])
print("Total samples:", seismic["sample_count"])
print("Displayed traces:", seismic["displayed_traces"])
print("Amplitude matrix:", seismic["amplitudes"].shape)

# Generate seismic figure
print("\nGenerating seismic section...")

fig = plot_seismic_section(seismic)

# Export interactive figure
fig.write_html("full_stack_seismic.html")

print("\nSeismic section generated successfully!")
print("Output: full_stack_seismic.html")
