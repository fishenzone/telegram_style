from pathlib import Path

MODEL_NAME = "unsloth/Qwen3-14B"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
FULL_FINETUNING = False

TRAIN_SIZE = 20
NUM_EPOCHS = 3
SEED = 66

LORA_R = 16
LORA_ALPHA = 16

GEN_TEMPERATURE = 0.5
GEN_TOP_P = 0.8
GEN_TOP_K = 20

GEN_MAX_NEW_TOKENS_TYPE1 = 120
GEN_MAX_NEW_TOKENS_TYPE2 = 180

ROOT = Path(".")

ARTIFACTS_DIR = ROOT / "artifacts"
ADAPTERS_DIR = ROOT / "adapters"
REPORTS_DIR = ROOT / "reports"

RAW_INPUTS_BANKI = ROOT / "inputs_banki_oil.txt"
RAW_OUTPUTS_BANKI = ROOT / "outputs_banki_oil.txt"
RAW_INPUTS_MOSCOW = ROOT / "inputs_moscowach.txt"
RAW_OUTPUTS_MOSCOW = ROOT / "outputs_moscowach.txt"

INPUTS_TYPE1_PATH = ROOT / "inputs_type1.txt"
INPUTS_TYPE2_PATH = ROOT / "inputs_type2.txt"

OUTPUTS_TYPE1_PATH = ROOT / "outputs_type1.txt"
OUTPUTS_TYPE2_PATH = ROOT / "outputs_type2.txt"

BASELINE_TYPE1_PATH = ROOT / "baseline_type1.txt"
BASELINE_TYPE2_PATH = ROOT / "baseline_type2.txt"

REFERENCE_TYPE1_PATH = ROOT / "reference_type1.txt"
REFERENCE_TYPE2_PATH = ROOT / "reference_type2.txt"

TRAIN_PAIRS_TYPE1_PATH = ROOT / "train_pairs_type1.jsonl"
TRAIN_PAIRS_TYPE2_PATH = ROOT / "train_pairs_type2.jsonl"
TEST_PAIRS_TYPE1_PATH = ROOT / "test_pairs_type1.jsonl"
TEST_PAIRS_TYPE2_PATH = ROOT / "test_pairs_type2.jsonl"

FORMATTED_TRAIN_TYPE1_PATH = ROOT / "formatted_train_type1.txt"
FORMATTED_TRAIN_TYPE2_PATH = ROOT / "formatted_train_type2.txt"

LORA_TYPE1_DIR = ROOT / "lora_type1"
LORA_TYPE2_DIR = ROOT / "lora_type2"

METRICS_SUMMARY_CSV = ROOT / "metrics_summary.csv"
METRICS_STRUCTURE_CSV = ROOT / "metrics_structure.csv"
METRICS_CROSS_STYLE_CSV = ROOT / "metrics_cross_style.csv"

PLOT_COSINE_PATH = ROOT / "plot_cosine_similarity.png"
PLOT_STYLE_SCORE_PATH = ROOT / "plot_style_compliance.png"
PLOT_MARGIN_PATH = ROOT / "plot_style_margin.png"
PLOT_CROSS_STYLE_PATH = ROOT / "plot_cross_style_heatmap.png"

RESULTS_DRAFT_PATH = ROOT / "RESULTS_draft.md"

CHANNELS = {
    "type1": {
        "name": "banki_oil",
        "raw_inputs": RAW_INPUTS_BANKI,
        "raw_outputs": RAW_OUTPUTS_BANKI,
        "train_pairs_path": TRAIN_PAIRS_TYPE1_PATH,
        "test_pairs_path": TEST_PAIRS_TYPE1_PATH,
        "inputs_path": INPUTS_TYPE1_PATH,
        "reference_path": REFERENCE_TYPE1_PATH,
        "baseline_path": BASELINE_TYPE1_PATH,
        "outputs_path": OUTPUTS_TYPE1_PATH,
        "formatted_train_path": FORMATTED_TRAIN_TYPE1_PATH,
        "lora_dir": LORA_TYPE1_DIR,
        "max_new_tokens": GEN_MAX_NEW_TOKENS_TYPE1,
    },
    "type2": {
        "name": "moscowach",
        "raw_inputs": RAW_INPUTS_MOSCOW,
        "raw_outputs": RAW_OUTPUTS_MOSCOW,
        "train_pairs_path": TRAIN_PAIRS_TYPE2_PATH,
        "test_pairs_path": TEST_PAIRS_TYPE2_PATH,
        "inputs_path": INPUTS_TYPE2_PATH,
        "reference_path": REFERENCE_TYPE2_PATH,
        "baseline_path": BASELINE_TYPE2_PATH,
        "outputs_path": OUTPUTS_TYPE2_PATH,
        "formatted_train_path": FORMATTED_TRAIN_TYPE2_PATH,
        "lora_dir": LORA_TYPE2_DIR,
        "max_new_tokens": GEN_MAX_NEW_TOKENS_TYPE2,
    },
}
