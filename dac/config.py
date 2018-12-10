from pathlib2 import Path

dac_dir = Path(__file__).parent
trajector_dir = dac_dir / 'trajs'
project_dir = dac_dir.parent.parent
results_dir = project_dir / 'results'

if not trajector_dir.is_dir():
    trajector_dir.mkdir()

if not results_dir.is_dir():
    results_dir.mkdir()
