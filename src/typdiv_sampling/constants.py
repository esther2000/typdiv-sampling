from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EVAL_PATH = PROJECT_ROOT / "evaluation"
FRAME_PATH = DATA_PATH / "frames"

DEFAULT_DIST_PATH = DATA_PATH / "gb_lang_dists.csv"
DEFAULT_GB_PATH = PROJECT_ROOT / "grambank/cldf/languages.csv"
DEFAULT_WALS_PATH = DATA_PATH / "wals_dedup.csv"
DEFAULT_COUNTS_PATH = DATA_PATH / "convenience/convenience_counts.json"

DEFAULT_FRAME_PATH = FRAME_PATH / "gb_frame-bnrc75.txt"

DEFAULT_GB_FEATURES_PATH = DATA_PATH / "gb_processed.csv"
DEFAULT_DISTANCES_PATH = DATA_PATH / "gb_lang_dists.csv"
