"""Ask JEV the VSM 2013 items as matched Saudi and American personas, in English and Arabic.

JEV (TypeSafe AI) returns a probability for every answer option instead of generating text. Each of the
1,200 prompts in prompts.csv (25 persona conditions x 2 languages x 24 items) is sent 30 times in one of
eight request set-ups (Table 1 of the paper):

  1 documented      the persona as a named field of the state, the item wording with a note; Choice question
  2 llm_style       the persona system prompt as the state, the user prompt as the question; Choice question
  3 no_format       as 2, without the "IMPORTANT: Respond with ONLY ..." sentence
  4 score           as 2, the five options as the ordered levels of a Score question
  5 score_reversed  as 4, with the levels in reverse order
  6 third_person    the respondent described in the state; "Which answer would they choose?"
  7 batched         as 2, the 24 items of a persona and language as 24 questions in one request
  8 other_language  as 2, with the system prompt of the other language ("language" = language of the item)

Answers are appended to data/<set-up>.jsonl, one line per answer; a rerun skips answers already collected.

    export TYPESAFE_API_KEY=...                       # your own API key
    python run_jev.py --setup documented --dry-run --show 1
    python run_jev.py --setup documented
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

HERE = Path(__file__).parent
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
SETUPS = ("documented", "llm_style", "no_format", "score", "score_reversed", "third_person", "batched", "other_language")
USD_PER_INPUT_TOKEN = 0.042 / 1e6      # output tokens are free
NO_PERSONA = "baseline (no persona)"
OPTION = re.compile(r"^\s*([1-5])\s*[.)]\s*(.+?)\s*$", re.M)
FORMAT_SENTENCE = re.compile(r"\s*(IMPORTANT:|هام:).*$", re.S)
ARABIC_PERSONA = re.compile(r"^تخيل أنك (.+?)\. ")
GENERIC_LEADS = ("Please answer the below question by choosing an answer from the following:",
                 "حدد درجة أهمية الموضوع أدناه مستخدما المعيار التالي:")
THIRD_PERSON = {   # state, description when there is no persona, lead-in of the question
    "English": ("Description of the survey respondent: {who}.", "not available",
                "This respondent was asked the question below. Which answer would they choose?\n\n"),
    "Arabic": ("وصف المشارك في الاستبيان: {who}.", "غير متوفر",
               "طُرح على هذا المشارك السؤال أدناه. ما الإجابة التي سيختارها؟\n\n"),
}
DOCUMENTED = {     # field name, value when there is no persona, note
    "English": ("respondent", "not described", "Answer as the respondent described in `respondent` would answer."),
    "Arabic": ("المشارك", "غير موصوف", "أجب كما سيجيب المشارك الموصوف في `المشارك`."),
}


def criteria(user_prompt):
    """The prompt's own answer options, keyed "1"-"5"."""
    found = {m.group(1): m.group(2).strip() for m in OPTION.finditer(user_prompt)}
    if sorted(found) != ["1", "2", "3", "4", "5"]:
        raise ValueError(f"expected options 1-5, found {sorted(found)}")
    return found


def persona_words(row, nobody):
    if row.persona == NO_PERSONA:
        return nobody
    return row.persona if row.language == "English" else ARABIC_PERSONA.match(row.system_prompt).group(1)


def question_only(user_prompt):
    """The item wording without the answer list and the generic "choose from the following" line."""
    lines = [line.strip() for line in user_prompt.split("\n")]
    return "\n".join(line for line in lines if line and not OPTION.match(line) and line not in GENERIC_LEADS)


def build_body(row, setup):
    """The request for one prompt in one set-up (the batched set-up is assembled in request_for)."""
    state, instructions = row.system_prompt, row.user_prompt
    options = criteria(row.user_prompt)
    if setup == "documented":
        field, nobody, note = DOCUMENTED[row.language]
        state = {field: persona_words(row, nobody)}
        instructions = {"question": question_only(row.user_prompt), "note": note}
    elif setup == "no_format":
        state = FORMAT_SENTENCE.sub("", state).strip()
    elif setup == "third_person":
        template, nobody, lead = THIRD_PERSON[row.language]
        state, instructions = template.format(who=persona_words(row, nobody)), lead + row.user_prompt
    elif setup == "other_language":
        state = row.other_system_prompt
    if setup in ("score", "score_reversed"):   # level k is option k + 1, or option 5 - k when reversed
        order = sorted(options, reverse=setup == "score_reversed")
        question = {"type": "score", "instructions": instructions, "criteria": [f"{k}. {options[k]}" for k in order]}
    else:
        question = {"type": "choice", "instructions": instructions, "criteria": options}
    return {"model": MODEL, "state": state, "questions": {"answer": question}}


def request_for(rows, setup):
    """The request body and the prompt behind each question key: one prompt per request, or all 24 items
    of a persona and language in one request when batched (each question is still answered on its own)."""
    if setup != "batched":
        return build_body(rows[0], setup), {"answer": rows[0]}
    body = build_body(rows[0], "llm_style")
    body["questions"] = {r.prompt_id: build_body(r, "llm_style")["questions"]["answer"] for r in rows}
    return body, {r.prompt_id: r for r in rows}


def read_answer(ans, setup):
    """Option probabilities keyed "1"-"5" and JEV's most probable option."""
    if ans["type"] == "score":
        option = (lambda level: 5 - level) if setup == "score_reversed" else (lambda level: level + 1)
        probs = {str(option(int(k))): float(v) for k, v in ans["probabilities"].items()}
        return probs, int(max(probs, key=probs.get))
    return {k: float(v) for k, v in ans["probabilities"].items()}, int(ans["choice"])


def record(row, iteration, setup, question_type):
    rec = {"setup": setup, "model": MODEL, "prompt_id": row.prompt_id, "iteration": iteration,
           "persona": row.persona, "target_country": row.target_country, "complexity": row.complexity,
           "language": row.language, "question_id": row.question_id, "question_type": question_type}
    if setup == "other_language":
        rec["framing_language"] = "English" if row.language == "Arabic" else "Arabic"
    return rec


async def call(client, key, rows, iteration, setup, sem, out, stats):
    body, by_key = request_for(rows, setup)
    records = {k: record(r, iteration, setup, body["questions"][k]["type"]) for k, r in by_key.items()}
    async with sem:
        for attempt in range(8):
            try:
                resp = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
                if resp.status_code == 429 or resp.status_code >= 500:     # rate limit or server error: retry
                    stats["429"] += resp.status_code == 429
                    raise httpx.HTTPStatusError(str(resp.status_code), request=resp.request, response=resp)
                if resp.status_code >= 400:                                  # other client errors: no retry
                    for rec in records.values():
                        rec.update(error=f"HTTP {resp.status_code}: {resp.text[:200]}")
                    break
                data = resp.json()
                parsed = {}
                for k in by_key:
                    ans = data["answers"][k]
                    probs, choice = read_answer(ans, setup)
                    if sum(probs.values()) <= 0:
                        raise ValueError("no probability mass returned")
                    parsed[k] = (ans, probs, choice)
                usage = data.get("usage") or {}
                stats["usd"] += usage.get("input_tokens", 0) * USD_PER_INPUT_TOKEN
                for k, (ans, probs, choice) in parsed.items():
                    records[k].update(choice=choice, probabilities=probs, confidence=ans.get("confidence"),
                                      expected=round(sum(int(o) * v for o, v in probs.items()) / sum(probs.values()), 4),
                                      resolved_model=data.get("model"), usage=usage, error=None)
                break
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                for rec in records.values():
                    rec.update(error=f"{type(exc).__name__}: {exc}"[:300])
                await asyncio.sleep(min(60, 2 ** attempt))
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for rec in records.values():
            rec["timestamp"] = stamp
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        stats["done"] += len(records)


def done_keys(path):
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {(r["prompt_id"], r["iteration"]) for r in rows if r.get("error") is None}


def load_prompts():
    prompts = pd.read_csv(HERE / "prompts.csv")
    system = prompts.drop_duplicates(["persona", "language"]).set_index(["persona", "language"]).system_prompt
    other = {"English": "Arabic", "Arabic": "English"}
    prompts["other_system_prompt"] = [system[(p, other[lang])] for p, lang in zip(prompts.persona, prompts.language)]
    return prompts


async def progress(stats, total, t0):
    while True:
        await asyncio.sleep(20)
        rate = stats["done"] / max(time.time() - t0, 1)
        print(f"  {stats['done']:,}/{total:,} answers, {rate:.1f}/s, ${stats['usd']:.3f}, 429s: {stats['429']}", flush=True)


async def main(args):
    prompts = load_prompts()
    for row in prompts.itertuples():           # fail fast on a prompt that does not convert
        build_body(row, "llm_style" if args.setup == "batched" else args.setup)
    if args.limit:
        prompts = prompts.head(args.limit)
    path = Path(args.data) / f"{args.setup}.jsonl"
    done = done_keys(path)
    todo = [(row, it) for row in prompts.itertuples() for it in range(args.iterations) if (row.prompt_id, it) not in done]
    groups = {}
    for row, it in todo:
        group = (row.persona, row.language, it) if args.setup == "batched" else (row.prompt_id, it)
        groups.setdefault(group, []).append(row)
    jobs = [(rows, group[-1]) for group, rows in groups.items()]
    print(f"{args.setup}: {len(prompts)} prompts x {args.iterations} runs -> {len(todo):,} answers in {len(jobs):,} requests")
    for rows, _ in jobs[:args.show]:
        body, _ = request_for(rows, args.setup)
        shown = dict(list(body["questions"].items())[:2])
        print(json.dumps({**body, "questions": shown}, ensure_ascii=False, indent=1)
              + (f"\n... {len(body['questions'])} questions in this request" if len(body["questions"]) > 2 else ""))
    if args.dry_run or not todo:
        return
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("Set the TYPESAFE_API_KEY environment variable to your TypeSafe API key.")
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"usd": 0.0, "429": 0, "done": 0}
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        with path.open("a", encoding="utf-8") as out:
            ticker = asyncio.create_task(progress(stats, len(todo), time.time()))
            await asyncio.gather(*(call(client, key, rows, it, args.setup, sem, out, stats) for rows, it in jobs))
            ticker.cancel()
    errors = len(todo) - len(done_keys(path) - done)
    print(f"finished: {len(todo) - errors:,} valid, {errors:,} errors, ${stats['usd']:.3f}, 429s: {stats['429']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=SETUPS, required=True)
    ap.add_argument("--iterations", type=int, default=30, help="calls per prompt (default 30)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--data", default=str(HERE / "data"), help="output folder (default data/)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (for testing)")
    ap.add_argument("--show", type=int, default=0, help="print the first N request bodies")
    ap.add_argument("--dry-run", action="store_true", help="build the requests without calling the API")
    asyncio.run(main(ap.parse_args()))
