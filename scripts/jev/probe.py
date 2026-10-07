"""Black-box probes of the Jev API (jev-1.13.0) for the Jev article.

Reads TYPESAFE_API_KEY from .env.local. Writes public/measurements/jev/jev-probes.json.
Usage: python3 scripts/jev/probe.py [accounting isolation order latency_questions latency_state options]
"""
import itertools, json, os, random, statistics, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/measurements/jev/jev-probes.json"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
TICKET = "My payouts have failed three times this week. The bank says everything is fine on their side. Can someone please fix this today?"


def api_key():
    for line in (ROOT / ".env.local").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("TYPESAFE_API_KEY missing in .env.local")


KEY = api_key()


def call(state, questions, attempts=5):
    body = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
                wall = (time.perf_counter() - t0) * 1000
                upstream = r.headers.get("x-envoy-upstream-service-time")
                return {"wall_ms": round(wall, 1), "upstream_ms": int(upstream) if upstream else None, "usage": data["usage"], "answers": data["answers"]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503) and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            return {"error": e.code, "body": e.read().decode()[:400]}
    return {"error": "retries"}


def noul(text):
    return {"type": "noul", "instructions": text}


def accounting():
    """Input tokens as questions are added: is the state counted once?"""
    q = "Does this message convey urgency?"
    rows = []
    for n in (1, 2, 3, 5, 10):
        r = call(TICKET, {f"q{i}": noul(q) for i in range(n)})
        rows.append({"questions": n, "usage": r.get("usage")})
    long_state = " ".join([TICKET] * 20)
    for n in (1, 2, 10):
        r = call(long_state, {f"q{i}": noul(q) for i in range(n)})
        rows.append({"state": "ticket x20", "questions": n, "usage": r.get("usage")})
    return rows


def isolation():
    """Can one question read another question's text?"""
    crit = {"ZEBRA-7741": None, "OTTER-1029": None, "FALCON-5530": None, "none": "No secret code is given anywhere"}
    probe = {"type": "choice", "instructions": "Which secret code is declared for this request?", "criteria": crit}
    secret = "The secret code for this request is ZEBRA-7741."
    weather = "The weather is nice today and the park is full of people."
    rows = []
    for _ in range(5):
        a = call(weather, {"sibling": noul(secret + " Is the weather described as nice?"), "probe": probe})
        b = call(weather + " " + secret, {"sibling": noul("Is the weather described as nice?"), "probe": probe})
        rows.append({"secret_in_sibling": a["answers"]["probe"]["probabilities"], "secret_in_state": b["answers"]["probe"]["probabilities"]})
    return rows


def order():
    """Same options, six orders."""
    opts = {"payments": "Payout failures and payment processing", "account": "Login and account access", "bank": "A problem on the customer's bank side"}
    ticket = "My payout has not arrived. I changed my bank details last week and I am not sure whether the change went through."
    rows = []
    for perm in itertools.permutations(opts):
        for _ in range(3):
            r = call(ticket, {"queue":{"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {k: opts[k] for k in perm}}})
            rows.append({"order": list(perm), "probabilities": r["answers"]["queue"]["probabilities"]})
    return rows


def timed(make, sizes, repeats):
    jobs = [(s, i) for s in sizes for i in range(repeats)]
    random.shuffle(jobs)
    raw = {s: [] for s in sizes}
    for s, _ in jobs:
        state, questions = make(s)
        r = call(state, questions)
        if "error" in r:
            raw[s].append(r)
        else:
            raw[s].append({"wall_ms": r["wall_ms"], "upstream_ms": r["upstream_ms"], "usage": r["usage"]})
    out = []
    for s in sizes:
        ok = [x for x in raw[s] if "wall_ms" in x]
        out.append({"size": s, "runs": raw[s], "median_wall_ms": statistics.median(x["wall_ms"] for x in ok) if ok else None,
                    "min_wall_ms": min((x["wall_ms"] for x in ok), default=None), "input_tokens": ok[0]["usage"]["input_tokens"] if ok else None})
    return out


def latency_questions():
    """Short state, 1..1500 distinct yes/no questions."""
    topics = ["urgency", "anger", "a refund request", "a payout failure", "a login problem", "a bank error", "a thank-you", "a legal threat"]
    make = lambda n: (TICKET, {f"q{i}": noul(f"Does this message mention {topics[i % len(topics)]}? (check {i})") for i in range(n)})
    return timed(make, [1, 10, 100, 300, 600, 1000, 1500], 6)


def latency_state():
    """One question, state from ~0.3k to ~30k tokens."""
    filler = "Ticket log entry: the customer wrote in about a delayed shipment and was told to wait two business days. "
    make = lambda k: (TICKET + " " + filler * k, {"q": noul("Does the first message convey urgency?")})
    return timed(make, [0, 80, 400, 800, 1250], 6)


def options():
    """One Choice with 2..200 options: does time follow the reported output tokens?"""
    make = lambda k: (TICKET, {"q": {"type": "choice", "instructions": "Which label fits this ticket best?",
                                     "criteria": {**{f"label_{i}": f"Unrelated category number {i}" for i in range(k - 1)}, "payments": "Payout failures"}}})
    rows = timed(make, [2, 20, 100, 200], 6)
    for row in rows:
        ok = [x for x in row["runs"] if "usage" in x]
        row["output_tokens"] = ok[0]["usage"]["output_tokens"] if ok else None
    return rows


PROBES = {"accounting": accounting, "isolation": isolation, "order": order, "latency_questions": latency_questions, "latency_state": latency_state, "options": options}

if __name__ == "__main__":
    names = sys.argv[1:] or list(PROBES)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["meta"] = {"model": MODEL, "date": time.strftime("%Y-%m-%d"), "note": "wall_ms is client-side round trip from the author's laptop; upstream_ms is the x-envoy-upstream-service-time header when present"}
    for name in names:
        print("==", name, flush=True)
        result[name] = PROBES[name]()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1))
        summary = [{k: v for k, v in row.items() if k != "runs"} for row in result[name]]
        print(json.dumps(summary, indent=1)[:3000], flush=True)
