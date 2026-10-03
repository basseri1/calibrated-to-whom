# Calibrated to Whom? Persona and Language Effects on Cultural Values in JEV

Code for the paper

> Bushra Asseri and Abdulaziz Asseri. 2026. *Calibrated to Whom? Persona and Language Effects on Cultural Values in JEV.* [arXiv:2609.36399](https://arxiv.org/abs/2609.36399).

JEV (TypeSafe AI) is a decision-only model: it returns a probability for every answer option instead of generating text. We asked it the 24 items of the Values Survey Module 2013 (VSM 2013) as 12 matched Saudi and 12 matched American personas and without a persona, in English and Arabic, under eight ways of formulating the request. Each of the 1,200 prompts was sent 30 times in each set-up, for 288,000 answers in total.

## Files

| File | Content |
|---|---|
| `prompts.csv` | The 1,200 prompts: 25 persona conditions × 2 languages × 24 items |
| `run_jev.py` | Sends the prompts to JEV in one of the eight set-ups and saves the answers |
| `analyze.py` | Computes the results reported in the paper |
| `vsm.py` | VSM 2013 scoring and the human reference scores |

## Setup

Python 3.10 or later:

```
pip install -r requirements.txt
```

Collecting answers needs your own TypeSafe API key, read from an environment variable:

```
export TYPESAFE_API_KEY=your-key
```

## Collecting the answers

```
python run_jev.py --setup documented --dry-run --show 1    # print one request, no API call
python run_jev.py --setup documented                        # 36,000 answers -> data/documented.jsonl
```

| # | `--setup` | Request |
|---|---|---|
| 1 | `documented` | Persona as a named field of the state; item wording and a note; Choice question (headline set-up) |
| 2 | `llm_style` | Persona system prompt as the state; user prompt as a Choice question |
| 3 | `no_format` | As 2, without "IMPORTANT: Respond with ONLY …" |
| 4 | `score` | As 2, with the options as the ordered levels of a Score question |
| 5 | `score_reversed` | As 4, with the levels in reverse order |
| 6 | `third_person` | Respondent described in the state; "Which answer would they choose?" |
| 7 | `batched` | As 2, with the 24 items of a persona and language in one request |
| 8 | `other_language` | As 2, with the system prompt in the other language |

The model is pinned to `jev-1.13.0`. A set-up costs about US$1 at US$0.042 per million input tokens (output tokens are free). A rerun skips the answers already collected. JEV has no seed or temperature parameter, so repeated calls can differ slightly, and a new run will give numbers close to, but not identical with, those in the paper.

Each line of `data/<set-up>.jsonl` is one answer: JEV's probability for each of the five options, its most probable option, its confidence, and the expected answer Σ k·p_k.

## Analysis

```
python analyze.py          # prints a summary and writes results.json
```

Every measure compares two profiles measured on the same scale, such as Saudi and American personas or a profile and the no-persona default, so the additive constants in the VSM formulas cancel. The 12 matched persona pairs are the units of replication; intervals are t-intervals or bootstrap intervals over these pairs (4,000 resamples, seed 2026). Run on the answers collected for the paper, `analyze.py` reproduces the numbers reported in it.

## Questionnaire

The English prompts use the wording of the VSM 2013 and the Arabic prompts its official Arabic translation (Hofstede & Minkov, 2013; [geerthofstede.com](https://geerthofstede.com)). The license below covers the code, not the questionnaire.

## Citation

```bibtex
@misc{asseri2026calibrated,
  title         = {Calibrated to Whom? Persona and Language Effects on Cultural Values in {JEV}},
  author        = {Asseri, Bushra and Asseri, Abdulaziz},
  year          = {2026},
  eprint        = {2609.36399},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License

MIT (code). See `LICENSE`.
