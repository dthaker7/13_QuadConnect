# QuadConnect: multi-model system design (P1-A7)

**The Connectors · Team 13 · INFO 490, Fall 2026**

This document holds the architectural reasoning behind QuadConnect's AI features. The experiments behind every
number are in [`multi_model_system.ipynb`](multi_model_system.ipynb), and every model reply is in
[`results/`](results/).

| Part | Section | Status |
|---|---|---|
| 1. Model size selection and benchmarking | [1. Which models, for which jobs](#1-which-models-for-which-jobs) | Done |
| 2. Cost comparison and infrastructure | 2. Cost at six traffic levels | Next |
| 3. Multi-model routing | 3. Routing strategies | Next |

---

## 1. Which models, for which jobs

### What we measured

The 15 models from P1-A6, three in each size category, each answered the same **25 prompts**: five prompt types
QuadConnect needs, five realistic inputs each.

| Prompt type | QuadConnect's job |
|---|---|
| Structured extraction | Turn a search request into filters from QuadConnect's own lists |
| Classification | Approve, review or reject a suggested meeting place |
| Prediction | Read a post-meetup note: the star rating, and whether the student wants to stay connected |
| Summarization | One sentence for staff from three feedback notes, with no names |
| Text generation | Three icebreaker questions about a new match's shared interests |

Every reply was scored by the same rules in code, from 1 to 10, on accuracy against a reference answer, instruction
adherence, structure and clarity. Accuracy gates the score: a wrong answer scores 1 however neat it looks. In a hand
check of 20 random replies, the code differed from our own scores by 0.94 points on average.

Twelve models ran on our laptop's CPU (an AMD Ryzen 7 6800HS: float32 under 2B, 4-bit NF4 from 2B up), and the
three over 15B ran on Groq's API.

### Results

| Model | Category | Mean score (1–10) | Median latency | Tokens/sec |
|---|---|---|---|---|
| SmolLM2-135M | Ultra-Light | 4.51 | 2.3 s | 17.7 |
| SmolLM2-360M | Ultra-Light | 6.81 | 3.9 s | 9.1 |
| Qwen2.5-0.5B | Ultra-Light | 6.66 | 4.4 s | 8.1 |
| Qwen2.5-1.5B | Small | 7.77 | 11.6 s | 3.4 |
| Falcon3-1B | Small | 7.25 | 9.4 s | 4.2 |
| granite-3.1-2b | Small | 9.11 | 39.8 s | 0.9 |
| **Qwen2.5-3B** | Medium | **9.32** | 34.9 s | 0.8 |
| Falcon3-3B | Medium | 8.34 | 32.2 s | 0.9 |
| Phi-4-mini | Medium | 9.19 | 46.1 s | 0.6 |
| Qwen2.5-7B | Large | 9.27 | 85.9 s | 0.3 |
| Llama-3.1-Tulu-3-8B | Large | 9.19 | 89.7 s | 0.3 |
| Falcon3-10B | Large | 9.57 | 138.0 s | 0.2 |
| gpt-oss-20b | Very Large (Groq) | 9.85 | 0.50 s | 921 |
| **Qwen3.8-27B** | Very Large (Groq) | **9.96** | 0.46 s | 490 |
| gpt-oss-120b | Very Large (Groq) | 9.96 | 0.55 s | 476 |

Local latency is our CPU; hosted latency is Groq's hardware plus the network, so the two are not comparable.

**By category and prompt type (mean score):**

| Category | Extraction | Classification | Prediction | Summarization | Text generation | All |
|---|---|---|---|---|---|---|
| Ultra-Light (< 1B) | 3.75 | 3.96 | 6.04 | 7.30 | 8.91 | 5.99 |
| Small (1B–3B) | 5.33 | 7.60 | 8.32 | 8.97 | 10.00 | 8.04 |
| Medium (3B–7B) | 7.19 | 8.80 | 9.46 | 9.32 | 9.97 | 8.95 |
| Large (7B–15B) | 8.64 | 9.40 | 9.64 | 9.04 | 10.00 | 9.34 |
| Very Large (> 15B) | 9.80 | 10.00 | 9.82 | 10.00 | 10.00 | 9.92 |

### What this means for the architecture

1. **The jobs are not equally hard, so one model is the wrong unit.**
   - Icebreakers are solved from 1.5B up: 11 of the 15 models score a perfect 10.
   - Extraction needs exact values from long lists. That is where small models fail (26 of the 75 extraction
     replies used a value not on the list), and only the Very Large category clears 9.5 on it.
   - So the system should send each job to the smallest model that does it well, not every job to one model.
2. **Bigger is better on average, but not model by model.** Size and score have a rank correlation of 0.94, yet
   Qwen2.5-3B (9.32) beat three larger local models, and granite-3.1-2b beat Falcon3-3B. The choice of model inside a
   tier should come from measurement, not from parameter count.
3. **Three candidate tiers come out of Part 1.** These are the building blocks for the cost model (Part 2) and the
   routing (Part 3):

   | Tier | Model | Why |
   |---|---|---|
   | Fast local | Qwen2.5-1.5B | Answers in about 12 s on a laptop CPU; on a GPU, A6 estimated 0.4 s (an NVIDIA L4). Perfect on icebreakers, 9.2 on summaries. It is the model QuadConnect's search already runs (A6). |
   | Quality local | Qwen2.5-3B | The best local balance: 9.32, within 0.25 of the best local model at a quarter of its latency. It needs a GPU to be interactive. |
   | API | Qwen3.8-27B on Groq (gpt-oss-20b as a second provider) | 9.96 in under half a second. It costs per token, which Part 2 prices. |

4. **Checks stay outside the models.** Larger models still broke rules:
   - 10 of the 12 local models, up to 10B, put a participant's name in a staff summary.
   - Falcon3-10B leaked a chat-template token into a reply.
   - Two Large models missed a request for staff to follow up.

   Whatever the routing, output checks stay in front of the user: the allow-list and strict schema QuadConnect's
   search already applies, and a no-names check for any summary.
5. **On our own hardware, the CPU is the bottleneck.** The 4-bit models generate under 1 token per second on this
   CPU (0.2 to 0.9), against 3 to 18 for the small full-precision models. Local hosting at scale means GPUs; Part 2
   compares that with paying per token.
