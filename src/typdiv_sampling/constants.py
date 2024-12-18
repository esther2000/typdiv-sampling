from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EVAL_PATH = PROJECT_ROOT / "evaluation"
FRAME_PATH = DATA_PATH / "frames"

DEFAULT_GB_LANGUAGES_PATH = PROJECT_ROOT / "grambank/cldf/languages.csv"
DEFAULT_WALS_PATH = DATA_PATH / "wals_dedup.csv"
DEFAULT_COUNTS_PATH = DATA_PATH / "convenience/convenience_counts.json"

DEFAULT_ANNOTATIONS_PATH = DATA_PATH / "annotations-enhanced.csv"

DEFAULT_LANGUOID_PATH = DATA_PATH / "languoid.csv"

DEFAULT_FRAME_PATH = FRAME_PATH / "gb_frame-bnrc75.txt"

DEFAULT_GB_RAW_FEATURES_PATH = DATA_PATH / "gb_lang_feat_vals.csv"
DEFAULT_GB_FEATURES_PATH = DATA_PATH / "gb_processed.csv"
DEFAULT_DISTANCES_PATH = DATA_PATH / "gb_lang_dists.csv"

GB_MULTI_VALUE_FEATURES = ["GB024", "GB025", "GB065", "GB130", "GB193", "GB203"]
