"""Statistical check of vLLM v0.30.0's rejection sampler on a 4-token vocabulary.

Feeds the real `rejection_sample` (Triton kernels) N independent one-draft-token requests and
compares the empirical distribution of the first output token with the target distribution p.
Only `all_greedy`, `all_random`, `temperature` and `generators` of SamplingMetadata are read by
this function, so a namespace with those fields stands in for the full dataclass.
"""
import json, sys, types
import torch
from vllm.v1.sample.rejection_sampler import rejection_sample, PLACEHOLDER_TOKEN_ID

dev = "cuda"
p = torch.tensor([0.1, 0.6, 0.2, 0.1], device=dev)   # same distributions as Figure 1, position 3
q = torch.tensor([0.4, 0.2, 0.3, 0.1], device=dev)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8_000_000
V = 4
meta = types.SimpleNamespace(all_greedy=False, all_random=True, temperature=torch.ones(N, device=dev), generators={})

def run(draft_ids, draft_probs, seed):
    torch.manual_seed(seed)
    out = rejection_sample(
        draft_token_ids=draft_ids.to(torch.int32).contiguous(),
        num_draft_tokens=[1] * N,
        max_spec_len=1,
        cu_num_draft_tokens=torch.arange(1, N + 1, device=dev, dtype=torch.int32),
        draft_probs=draft_probs,
        target_logits=p.log().expand(N, V).contiguous(),
        bonus_token_ids=torch.zeros((N, 1), dtype=torch.int32, device=dev),
        sampling_metadata=meta,
    )
    first = out[:, 0].long()
    accepted = out[:, 1] != PLACEHOLDER_TOKEN_ID   # bonus token present <=> draft token accepted
    hist = torch.bincount(first, minlength=V).float() / N
    return hist.cpu().tolist(), accepted.float().mean().item()

res = {"N": N, "p": p.tolist(), "q": q.tolist(), "alpha_theory": torch.minimum(p, q).sum().item()}
# A: draft samples from q, full q provided ('probabilistic')
draft = torch.multinomial(q, N, replacement=True)
hist, acc = run(draft, q.expand(N, V).contiguous(), 1)
res["prob_draft"] = {"empirical_first_token": hist, "accept_rate": acc}
# B: one-hot q (draft_probs=None, the 'greedy' draft path), each fixed draft token
res["onehot_draft"] = {}
for d in range(V):
    hist, acc = run(torch.full((N,), d, device=dev), None, 10 + d)
    res["onehot_draft"][str(d)] = {"empirical_first_token": hist, "accept_rate": acc, "accept_theory": p[d].item()}
# C: wrong rule for contrast, sampling p directly on reject would give
res["tv_prob"] = 0.5 * sum(abs(a - b) for a, b in zip(res["prob_draft"]["empirical_first_token"], res["p"]))
res["tv_onehot"] = {d: 0.5 * sum(abs(a - b) for a, b in zip(v["empirical_first_token"], res["p"])) for d, v in res["onehot_draft"].items()}
res["tv_noise_scale"] = (V / (2 * 3.14159 * N)) ** 0.5  # rough size of sampling noise in TV
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "kernel_check.json", "w"), indent=1)
