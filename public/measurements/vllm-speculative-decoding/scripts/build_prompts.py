"""Build the two workloads: chat (open-ended questions) and summ (wikitext-2 passages)."""
import json, re, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL = sys.argv[1]
PARQUET = sys.argv[2]
OUT = sys.argv[3]
tok = AutoTokenizer.from_pretrained(MODEL)

topics = ["how a CPU cache works", "why the sky is blue", "the difference between TCP and UDP",
          "how vaccines train the immune system", "what causes inflation", "how a hash table handles collisions",
          "why we need sleep", "how GPS determines position", "what a transformer model does",
          "how photosynthesis works", "the causes of the French Revolution", "how public-key encryption works",
          "why the ocean has tides", "what makes a good unit test", "how compilers optimize loops",
          "the basics of supply and demand"]
forms = ["Explain {t} to a curious high-school student.",
         "Give a concise technical overview of {t}, with one concrete example.",
         "Write a short blog-style paragraph about {t} and mention one common misconception."]
chat_q = [f.format(t=t) for t in topics for f in forms]  # 48

def render(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

table = pq.read_table(PARQUET).column("text").to_pylist()
paras = [t.strip() for t in table if len(t.split()) > 120 and not t.strip().startswith("=")]
summ = []
for p in paras:
    p = re.sub(r"\s+([,.;:!?])", r"\1", p.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", "."))
    n = len(tok(p).input_ids)
    if 250 <= n <= 450:
        summ.append(p)
    if len(summ) == 48:
        break
assert len(summ) == 48, len(summ)
out = {
    "chat": [render(q) for q in chat_q],
    "summ": [render("Summarize the following passage in about 100 words.\n\n" + p) for p in summ],
}
for k, v in out.items():
    lens = [len(tok(x).input_ids) for x in v]
    print(k, len(v), "input tokens min/mean/max", min(lens), sum(lens) / len(lens), max(lens))
json.dump(out, open(OUT, "w"), ensure_ascii=False)
