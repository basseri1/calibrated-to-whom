"""VSM 2013 scoring (Hofstede & Minkov, 2013) and the human reference scores used in the paper."""
import pandas as pd

DIMS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

# Hofstede, Hofstede & Minkov (2010) country scores, and the Saudi VSM 2013 scores of Almutairi et al. (2021).
REFERENCE = {"KSA": [95, 25, 60, 80, 36, 52],
             "USA": [40, 91, 62, 46, 26, 68],
             "KSA_almutairi": [72, 48, 43, 64, 27, 14]}


def indices(m, c=50.0):
    """The six VSM 2013 indices from the mean answers m[1]..m[24].

    Each index contains a constant C chosen by the researcher. Every measure in the paper compares two
    profiles, so C cancels; 50 is an arbitrary choice."""
    return {
        "PDI": 35 * (m[7] - m[2]) + 25 * (m[20] - m[23]) + c,
        "IDV": 35 * (m[4] - m[1]) + 35 * (m[9] - m[6]) + c,
        "MAS": 35 * (m[5] - m[3]) + 35 * (m[8] - m[10]) + c,
        "UAI": 40 * (m[18] - m[15]) + 25 * (m[21] - m[24]) + c,
        "LTO": 40 * (m[13] - m[14]) + 25 * (m[19] - m[22]) + c,
        "IVR": 35 * (m[12] - m[11]) + 40 * (m[17] - m[16]) + c,
    }


def score_profiles(long, answer="expected"):
    """One row of six indices per persona condition and language, from one row per answer.

    `answer` is the per-answer value that is averaged per item: JEV's expected answer sum_k k * p_k."""
    keys = ["persona", "target_country", "complexity", "language"]
    long = long.assign(item=long["question_id"].str[1:].astype(int))
    means = long.groupby(keys + ["item"])[answer].mean().unstack("item")
    return pd.DataFrame([indices(row) for _, row in means.iterrows()], index=means.index).reset_index()
