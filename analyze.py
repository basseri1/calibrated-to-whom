"""The paper's results from the answers collected by run_jev.py.

    python analyze.py                      # reads data/<set-up>.jsonl, prints a report, writes results.json

Every measure compares two of JEV's own profiles (or two human profiles), so the VSM constants cancel.
Repeated calls of a prompt are not independent observations: the 12 matched Saudi/American persona pairs
are the units of replication. Bootstrap intervals use 4,000 resamples of these pairs from one random stream
(seed 2026), drawn in the order of compute() below.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from run_jev import SETUPS
from vsm import DIMS, REFERENCE, score_profiles

HERE = Path(__file__).parent
HEADLINE = "documented"
LANGS = ("English", "Arabic")
USA = np.array(REFERENCE["USA"], dtype=float)
HUMAN = {"Hofstede": np.array(REFERENCE["KSA"]) - USA,            # Saudi minus US, per dimension
         "Almutairi": np.array(REFERENCE["KSA_almutairi"]) - USA}
H = HUMAN["Hofstede"]
B = 4000
RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------- data and profiles

def load(path):
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([r for r in map(json.loads, f) if r.get("error") is None])


def parse(persona):
    country = "KSA" if "Saudi" in persona else "USA" if "United States" in persona else None
    gender = "female" if "female" in persona else "male" if re.search(r"\bmale\b", persona) else "none"
    age = ("young" if "young" in persona else "middle aged" if "middle aged" in persona
           else "elder" if "elder" in persona else "none")
    profile = re.sub(r"\s*from (Saudi Arabia|United States)$", "", persona) or "(country only)"
    return country, gender, age, profile


def profiles(long):
    """Six indices per persona condition and language, with the persona's country, gender, age and profile."""
    cells = score_profiles(long)
    parsed = cells.persona.map(parse)
    for i, col in enumerate(("country", "gender", "age", "profile")):
        cells[col] = parsed.str[i]
    return cells


# ---------------------------------------------------------------- measures

def reliability(long):
    """Test-retest over the calls of each prompt, and the shape of JEV's answer distributions."""
    g = long.groupby("prompt_id").expected
    k, grand = g.size().mean(), long.expected.mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (g.ngroups - 1)
    msw = ((long.expected - g.transform("mean")) ** 2).sum() / (len(long) - g.ngroups)
    p = np.array([[d.get(str(o), 0.0) for o in range(1, 6)] for d in long.probabilities])
    p = p / p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1) / np.log(5)
    return {"same_top_choice_30_calls": (long.groupby("prompt_id").choice.nunique() == 1).mean(),
            "icc_expected": (msb - msw) / (msb + (k - 1) * msw),
            "sd_across_calls": g.std().mean(), "mean_confidence": long.confidence.mean(),
            "mean_norm_entropy": entropy.mean()}


def matched_national(cells, lang):
    """Saudi-minus-US difference for each of the 12 demographic profiles (profiles x dimensions)."""
    c = cells[(cells.language == lang) & cells.country.notna()]
    piv = c.pivot_table(index="profile", columns="country", values=DIMS)
    return np.stack([(piv[(d, "KSA")] - piv[(d, "USA")]).values for d in DIMS], axis=1)


def share_cos(mean_diff, human=H):
    """Share of the human difference reproduced along its direction (1 = same size), and direction cosine."""
    return mean_diff @ human / (human @ human), mean_diff @ human / (np.linalg.norm(mean_diff) * np.linalg.norm(human))


def boot(diffs_list, stat):
    """Bootstrap over matched profiles (the same resampled profiles for every array in diffs_list)."""
    n = diffs_list[0].shape[0]
    idx = RNG.integers(0, n, (B, n))
    return np.array([stat(*[d[i].mean(axis=0) for d in diffs_list]) for i in idx])


def default_position(cells, lang):
    """Where the no-persona profile lies between JEV's own American (0) and Saudi (1) personas."""
    c = cells[cells.language == lang]
    s = c[c.country == "KSA"][DIMS].mean().values
    u = c[c.country == "USA"][DIMS].mean().values
    b = c[c.target_country == "none"][DIMS].mean().values
    return (b - u) @ (s - u) / ((s - u) @ (s - u))


def pair_diffs(cells, attr, a, b, match):
    """a-minus-b differences for personas matched on country, language and the other attribute."""
    rows = []
    c = cells[cells.country.notna()]
    for (country, lang), g in c.groupby(["country", "language"]):
        ga, gb = g[g[attr] == a].set_index(match), g[g[attr] == b].set_index(match)
        for key in ga.index.intersection(gb.index):
            rows.append({"country": country, "language": lang, "matched_on": key,
                         **{d: ga.loc[key, d] - gb.loc[key, d] for d in DIMS}})
    return pd.DataFrame(rows)


def by_complexity(cells, lang):
    """Share and cosine for country-only personas (uni) and personas with one (bi) or two (tri) more attributes."""
    out = {}
    for level in ("uni", "bi", "tri"):
        c = cells[(cells.language == lang) & (cells.complexity == level)]
        out[level] = share_cos(c[c.country == "KSA"][DIMS].mean().values - c[c.country == "USA"][DIMS].mean().values)
    return out


def confidence_pairs(long, split, a, b, keys):
    """Mean confidence difference a-minus-b over prompts matched on `keys` (per-prompt means over calls)."""
    m = long.groupby(keys + [split]).confidence.mean().unstack(split).dropna()
    d = m[a] - m[b]
    return d.mean(), (d > 0).mean(), st.wilcoxon(d).pvalue, len(d)


def rank_order(diff, human=H):
    """Rank test of Masoud et al. (2025) for two countries: per dimension, whether JEV orders Saudi Arabia and
    the US as the human scores do (Kendall's tau +1) or not (-1)."""
    return np.sign(diff) == np.sign(human)


def language_default(cells):
    """Their model-level comparison: no persona, Arabic stands for Saudi Arabia and English for the US."""
    base = cells[cells.target_country == "none"].groupby("language")[DIMS].mean()
    return (base.loc["Arabic"] - base.loc["English"]).values


def t_ci(x):
    x = np.asarray(x, dtype=float)
    half = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def interval(point, bs):
    """Point estimate and 95% percentile interval, in percent."""
    return [round(100 * point, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def with_profile(long):
    persona_rows = long[long.target_country != "none"]
    return persona_rows.assign(profile=persona_rows.persona.map(lambda p: parse(p)[3]))


# ---------------------------------------------------------------- all results

def compute(data_dir):
    data = {s: load(Path(data_dir) / f"{s}.jsonl") for s in SETUPS if (Path(data_dir) / f"{s}.jsonl").exists()}
    if not data:
        raise SystemExit(f"no data/<set-up>.jsonl files in {data_dir}; run run_jev.py first")
    cells = {s: profiles(long) for s, long in data.items()}
    out = {"answers": {s: int(len(d)) for s, d in data.items()}}

    # language cross: the LLM-style prompt and its other-language framing (the bootstrap draws start here)
    if {"llm_style", "other_language"} <= set(cells):
        m = {lang: matched_national(cells["llm_style"], lang) for lang in LANGS}
        c = {lang: matched_national(cells["other_language"], lang) for lang in LANGS}
        cell = {("English", "English"): m["English"], ("English", "Arabic"): c["Arabic"],    # (framing, item)
                ("Arabic", "English"): c["English"], ("Arabic", "Arabic"): m["Arabic"]}
        cross = {}
        for frame in LANGS:
            for item in LANGS:
                d = cell[(frame, item)]
                cross[f"framing {frame} / item {item}"] = interval(share_cos(d.mean(axis=0))[0],
                                                                   boot([d], lambda mm: share_cos(mm)[0]))
        four = [cell[k] for k in (("English", "English"), ("English", "Arabic"), ("Arabic", "English"), ("Arabic", "Arabic"))]
        effects = {"item_effect": lambda ee, ea, ae, aa: (share_cos(ea)[0] - share_cos(ee)[0] + share_cos(aa)[0] - share_cos(ae)[0]) / 2,
                   "framing_effect": lambda ee, ea, ae, aa: (share_cos(ae)[0] - share_cos(ee)[0] + share_cos(aa)[0] - share_cos(ea)[0]) / 2}
        draws = {name: boot(four, f) for name, f in effects.items()}
        for name, f in effects.items():
            cross[name] = interval(f(*[d.mean(axis=0) for d in four]), draws[name])
        out["language_cross"] = cross

    if HEADLINE in cells:
        head, head_long = cells[HEADLINE], data[HEADLINE]
        diffs = {lang: matched_national(head, lang) for lang in LANGS}
        out["persona_detail"] = {lang: {lv: round(100 * v[0], 1) for lv, v in by_complexity(head, lang).items()}
                                 for lang in LANGS}
        summary = {}
        for ref, human in HUMAN.items():
            for lang in LANGS:
                s, cos = share_cos(diffs[lang].mean(axis=0), human)
                summary[f"{ref} / {lang}"] = {"share_pct": interval(s, boot([diffs[lang]], lambda mm: share_cos(mm, human)[0])),
                                              "cos": round(cos, 2)}
        gap = boot([diffs["Arabic"], diffs["English"]], lambda a, e: share_cos(a)[0] - share_cos(e)[0])
        summary["Arabic_minus_English_pct"] = interval(share_cos(diffs["Arabic"].mean(0))[0] - share_cos(diffs["English"].mean(0))[0], gap)
        out["national_summary"] = summary
        out["national_by_dimension"] = {"human": {k: [float(v) for v in h] for k, h in HUMAN.items()},
                                        "jev": {lang: {d: [round(v, 1) for v in t_ci(diffs[lang][:, j])] for j, d in enumerate(DIMS)}
                                                for lang in LANGS}}
        ga = {}
        for name, df in (("female_minus_male", pair_diffs(head, "gender", "female", "male", "age")),
                         ("elder_minus_young", pair_diffs(head, "age", "elder", "young", "gender"))):
            ga[name] = {}
            for country in ("KSA", "USA"):
                x = df[df.country == country]
                ga[name][country] = {d: [round(v, 1) for v in t_ci(x[d])] + [int((np.sign(x[d]) == np.sign(x[d].mean())).sum()), len(x)]
                                     for d in DIMS}
        out["gender_age"] = ga
        per = with_profile(head_long)
        mean = per.groupby(["question_id", "language", "profile", "target_country"]).expected.mean().unstack("target_country")
        item_diff = (mean["KSA"] - mean["USA"]).groupby(["question_id", "language"]).mean().unstack("language")
        item_conf = head_long.groupby(["question_id", "language"]).confidence.mean().unstack("language")
        out["items"] = {q: {"diff_EN": round(float(item_diff.loc[q, "English"]), 2), "diff_AR": round(float(item_diff.loc[q, "Arabic"]), 2),
                            "conf_EN": round(float(item_conf.loc[q, "English"]), 2), "conf_AR": round(float(item_conf.loc[q, "Arabic"]), 2)}
                        for q in item_diff.index}
        by_item = head_long.groupby("question_id").confidence.mean().sort_values()
        out["least_confident_items"] = {q: round(v, 2) for q, v in by_item.head(5).items()}
        out["most_confident_items"] = {q: round(v, 2) for q, v in by_item.tail(3).items()}

    if "llm_style" in cells:
        out["national_by_dimension_llm_style"] = {
            lang: {d: [round(v, 1) for v in t_ci(matched_national(cells["llm_style"], lang)[:, j])] for j, d in enumerate(DIMS)}
            for lang in LANGS}

    # by set-up
    out["reliability"] = {s: {k: round(float(x), 3) for k, x in reliability(data[s]).items()} for s in data}
    out["default_position"] = {s: {lang: round(default_position(cells[s], lang), 2) for lang in LANGS} for s in data}
    out["share_by_setup"], out["rank_agreement"], out["confidence"] = {}, {}, {}
    for s in data:
        d = {lang: matched_national(cells[s], lang) for lang in LANGS}
        out["share_by_setup"][s] = {lang: {"share_pct": round(100 * share_cos(d[lang].mean(0))[0], 1),
                                           "cos": round(share_cos(d[lang].mean(0))[1], 2)} for lang in LANGS}
        rank = {}
        for lang in LANGS:
            ok, per_profile = rank_order(d[lang].mean(0)), rank_order(d[lang])
            rank[lang] = {"in_order": int(ok.sum()), "mean_tau": round((2 * int(ok.sum()) - 6) / 6, 2),
                          "out_of_order": [dim for dim, o in zip(DIMS, ok) if not o],
                          "profiles_in_order": {dim: int(per_profile[:, j].sum()) for j, dim in enumerate(DIMS)},
                          "profiles_all_six": int(per_profile.all(axis=1).sum())}
        none = rank_order(language_default(cells[s]))
        rank["no_persona"] = {"in_order": int(none.sum()), "mean_tau": round((2 * int(none.sum()) - 6) / 6, 2),
                              "out_of_order": [dim for dim, o in zip(DIMS, none) if not o]}
        out["rank_agreement"][s] = rank
        a = confidence_pairs(data[s], "language", "English", "Arabic", ["persona", "question_id"])
        b = confidence_pairs(with_profile(data[s]), "target_country", "USA", "KSA", ["profile", "language", "question_id"])
        out["confidence"][s] = {name: {"mean_diff": round(x[0], 3), "share_positive": round(x[1], 2),
                                       "wilcoxon_p": float(f"{x[2]:.2g}"), "n_prompts": int(x[3])}
                                for name, x in (("English_minus_Arabic", a), ("American_minus_Saudi", b))}
    return out


def report(r):
    if "national_summary" in r:
        ns = r["national_summary"]
        print("Documented set-up (headline)")
        for ref in HUMAN:
            for lang in LANGS:
                x = ns[f"{ref} / {lang}"]
                print(f"  share of the {ref} Saudi-US difference, {lang}: {x['share_pct'][0]}% "
                      f"(95% CI {x['share_pct'][1]}-{x['share_pct'][2]}), cosine {x['cos']:.2f}")
        print(f"  Arabic minus English share (95% CI): {ns['Arabic_minus_English_pct']}")
        for lang in LANGS:
            print(f"  {lang}, Saudi minus American persona by dimension: " +
                  "; ".join(f"{d} {v[0]:+.1f}" for d, v in r["national_by_dimension"]["jev"][lang].items()))
        print(f"  persona detail (country only / + one / + two attributes): {r['persona_detail']}")
    if "language_cross" in r:
        print("Language cross (LLM-style prompt):")
        for k, v in r["language_cross"].items():
            print(f"  {k}: {v}")
    print("By set-up: share EN / AR, default position EN / AR, rank test in order EN / AR / no persona, ICC")
    for s in r["answers"]:
        sh, dp, rk = r["share_by_setup"][s], r["default_position"][s], r["rank_agreement"][s]
        print(f"  {s:15s} {sh['English']['share_pct']:6.1f}% {sh['Arabic']['share_pct']:6.1f}%   "
              f"{dp['English']:+.2f} {dp['Arabic']:+.2f}   {rk['English']['in_order']}/6 {rk['Arabic']['in_order']}/6 "
              f"{rk['no_persona']['in_order']}/6   ICC {r['reliability'][s]['icc_expected']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "data"), help="folder with <set-up>.jsonl files (default data/)")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    results = compute(args.data)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results)
    print(f"wrote {args.out}")
