from __future__ import annotations

FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FIELD_STRENGTH = {"0.1T": 0.1, "1.5T": 1.5, "3T": 3.0, "5T": 5.0, "7T": 7.0}
FIELD_TO_INDEX = {f: i for i, f in enumerate(FIELDS)}
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
MOD_TO_INDEX = {m: i for i, m in enumerate(MODALITIES)}

TASK_PAIRS = {
    1: [("0.1T", "7T"), ("1.5T", "7T"), ("3T", "7T"), ("5T", "7T")],
    2: [("0.1T", "1.5T"), ("0.1T", "3T"), ("0.1T", "5T"), ("0.1T", "7T")],
    3: [(s, t) for s in FIELDS for t in FIELDS if s != t],
}

# Validation IDs published in the current submission README. The scripts do not
# depend on these IDs, but they are useful for sanity checks and packaging logs.
VALIDATION_SOURCE_IDS = {
    "0.1T": ["0001", "0002", "0003"],
    "1.5T": ["0004", "0005", "0008"],
    "3T": ["0010", "0011", "0012"],
    "5T": ["0013", "0014", "0015"],
    "7T": ["0016", "0017", "0018"],
}
