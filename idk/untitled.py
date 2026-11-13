from pathlib import Path

project_dir = Path("/data/epic20/APGCE/apgce_team_04/idk")

print("Folder exists:", project_dir.exists())
print("Files:", [p.name for p in project_dir.iterdir()])
