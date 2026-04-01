# Grokking Phases - Agent Handoff

This document contains full context from the conversation that produced this experiment. A new Claude instance should read this before doing anything.

---

## What This Experiment Is

A **component-freezing ablation study on grokking**. A 4-layer transformer (820K params) is trained on modular division (a/b mod 97). At two checkpoints during the memorization plateau (step 7K and step 11K), each component is frozen and training continues for 10K more steps. The question: which components does grokking need, and does the answer change over time?

The finding: **yes, it changes.** Entry layer, exit MLP, and output head are needed at step 7K but dispensable at step 11K. Middle layers and weight decay are needed at both. This implies grokking has two sequential phases: infrastructure setup, then computational reorganization.

---

## Files In This Repo

| File | What it does |
|---|---|
| `grokking_full_metrics.py` | **Step 1**: Trains the base model on modular division for 30K steps. Saves checkpoints every 500 steps to `grokking_checkpoints.pt`. Also tracks geometric variables (r, alpha). |
| `grokking_ablation.py` | **Step 2**: Loads checkpoints from `grokking_checkpoints.pt`, freezes one component per run, trains 10K more steps, records trajectory. Saves results to `grokking_ablation_results.json`. |
| `grokking_full_metrics.json` | Full training metrics (30K steps): train_acc, test_acc, train_loss, test_loss, gap. |
| `grokking_math_vars_fine.json` | Geometric variables during training: r (=||delta||/||h||), alpha (=cos(h, delta)), and other derived quantities per step. Fine-grained around the grokking onset (100-step intervals from 10K-14K). |
| `grokking_ablation_results.json` | Full ablation results: 22 runs (11 interventions x 2 checkpoints). Each has intervention name, checkpoint, trainable/total params, final_test_acc, max_test_acc, and full trajectory. |
| `grokking_ablation_output.txt` | Raw terminal output from the ablation run. Empty (was run on a terminated RunPod server). |

**Important**: `grokking_checkpoints.pt` does NOT exist locally. It was on a RunPod server that's been terminated. To rerun the ablation, you must first run `grokking_full_metrics.py` to regenerate checkpoints. The JSON results from the original run are preserved.

---

## The Results

### Summary Table

| Intervention | From step 7K | From step 11K | Category |
|---|---|---|---|
| baseline | 92% | 97% | - |
| remove weight decay | 3% | 9% | always blocks |
| freeze middle layers (L1+L2) | 4% | 9% | always blocks |
| freeze all attention | 3% | 39% | always blocks |
| freeze all MLP | 6% | 13% | always blocks |
| freeze entry layer (L0) | 32% | 96% | phase-dependent |
| freeze exit MLP | 21% | 86% | phase-dependent |
| freeze output head | 41% | 95% | phase-dependent |
| freeze embeddings | 77% | 99% | never blocks |
| freeze exit attention | 67% | 65% | never blocks |

### CRITICAL DATA INTEGRITY ISSUE

The experiment ran each intervention for exactly 10K additional steps. But several results were **still rising at the cutoff**. This means "blocked" might actually be "delayed":

**From ckpt 7K - still rising when stopped:**
- `freeze_exit_mlp`: 21% and trajectory still climbing. Might grok with more steps.
- `freeze_exit_attn`: 67% and climbing. Would likely reach higher.
- `freeze_embed`: 77% and climbing (jumped from 25% to 77% in last 1K steps).
- `baseline`: 92% and still climbing (was at 76% just 500 steps before cutoff).

**From ckpt 11K - still rising when stopped:**
- `freeze_attn_all`: 39% but LAST point jumped from 17% to 39%. Accelerating. Given more steps it might grok. This would break the "always blocks" category.
- `freeze_middle_layers`: 9% but technically still rising (7.4% -> 8.6%). Probably flat.
- `freeze_mlp_all`: 13% and slowly rising.

**From ckpt 11K - PEAKED THEN DROPPED (instability):**
- `freeze_exit_layer`: peaked at 69.9% then dropped to 44.9%. Unstable.
- `freeze_exit_attn`: peaked at 97.3% then dropped to 64.8%. Hit near-perfect then collapsed.

**What this means**: The three-category classification (always blocks / phase-dependent / never blocks) assumes 10K steps is enough to distinguish "blocked" from "delayed." That assumption is not tested. The blog post needs to either:
1. Run longer (20-30K steps per intervention) to get definitive answers, OR
2. Explicitly caveat that "blocked" means "didn't grok in 10K steps" and some were trending up

This is the **most important thing to fix** before publishing.

---

## The Blog Post

### Location
`/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/content/blog/grokking-has-two-phases-and-you-can-see-the-boundary.mdx`

### Charts
6 SVG files in `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/content/blog/charts/`:
- `grok-curve.svg` - Classic grokking curve (train vs test accuracy, 30K steps)
- `grok-gap.svg` - Train-test gap over time
- `grok-7k-trajectories.svg` - All intervention trajectories from checkpoint 7K
- `grok-11k-trajectories.svg` - All intervention trajectories from checkpoint 11K
- `grok-phase-boundary.svg` - Side-by-side bar chart comparing 7K vs 11K final accuracy
- `grok-two-phases.svg` - Phase 1 (Infrastructure) vs Phase 2 (Computation) diagram

### Code Files
In `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/content/blog/code/`:
- `grokking_ablation.py` - Cleaned experiment code (embedded inline in post, collapsed with Copy/Download)
- `grokking_ablation_results.json` - Clean summary results (no internal paths/tokens)

### How Charts/Code Work in the Blog
The blog uses a **custom rehype plugin** at `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/src/lib/rehype-chart.ts` that processes two custom MDX elements at compile time:
- `<Chart src="filename.svg" />` - Reads SVG from `content/blog/charts/` and injects inline
- `<Code src="filename.py" title="..." collapsed />` - Reads code file from `content/blog/code/` and renders a collapsible dark code viewer with Copy + Download buttons

These work because they're processed at the rehype (HTML AST) level AFTER MDX parsing, so they bypass the angle-bracket-as-JSX problem. You CANNOT put raw HTML/SVG in MDX template literals or props - the MDX parser eats them.

### Citation Format
Inline: `[[N]](#ref-N)` - clickable link that jumps to reference
References section at the end with `<a id="ref-N"></a>` anchors before each entry.

### Blog Components
Defined in `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/src/components/MDXComponents.tsx`:
- `<Emphasis>` - Highlighted callout text
- `<Callout type="info|warning|tip">` - Boxed callout
- `<PullQuote>` - Centered pull quote
- `<DataView>` - Collapsible code/JSON viewer (registered but has the same MDX angle-bracket problem - use `<Code src="...">` via rehype instead)

### Blog Skills (for Claude Code in the blog repo)
Located in `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/.claude/skills/`:
- `write-blog/SKILL.md` - Voice, tone, formatting rules, code sharing rules, checklist
- `create-chart/SKILL.md` - SVG design system, colors, chart patterns, coordinate helpers
- `add-citations/SKILL.md` - Citation format, verification protocol, verified paper URLs

---

## Novelty Assessment

We did a thorough literature search. Here's where this stands:

### What's Already Known (~55-60% of the finding)
- Grokking has three phases: memorization, circuit formation, cleanup (Nanda et al. 2023, ICLR Oral)
- Weight decay is necessary (Power et al. 2022, confirmed many times)
- Embeddings converge early / are not the bottleneck (Xu et al. 2025, AlQuabeh et al. 2025)
- Component temporal ordering exists in single-layer models (Geometric Compression in Grokking, OpenReview)

### What's Genuinely Novel (~25-30%)
- **Checkpoint-dependent freezing as a methodology** - No prior work freezes components at checkpoint A vs B and compares. This experimental design is new.
- **The specific discrete phase boundary** - Entry layer, exit MLP, and head are needed at 7K but not at 11K. This operationally sharp finding is new.
- **Multi-layer decomposition** - Prior ablation work operates in Fourier space (Nanda) or single-layer models. The 4-layer entry/middle/exit decomposition is new.

### Publication Level
- Main venue (NeurIPS/ICML/ICLR): Probably not standalone. Delta is real but narrow.
- Workshop paper (MechInterp workshop): Yes, solid fit.
- Could become main venue if: (1) multiple seeds, (2) finer checkpoint sweep, (3) mechanistic analysis of WHAT changes at the boundary.

### Key Papers to Know
| Paper | arXiv | Relevance |
|---|---|---|
| Power et al. 2022 - Grokking | 2201.02177 | Original grokking paper |
| Nanda et al. 2023 - Progress Measures | 2301.05217 | Three-phase model, Fourier analysis |
| Liu et al. 2023 - Omnigrok | 2310.06110 | Weight decay + grokking beyond algorithmic data |
| Xu et al. 2025 - Let Me Grok for You | 2504.13292 | Embedding transfer eliminates delay |
| AlQuabeh et al. 2025 - Embedding Layer | 2505.15624 | MLPs without embeddings generalize immediately |
| Lyu et al. 2025 - Norm-Separation Delay | 2603.13331 | Theoretical: norm-driven phase transition |
| Geometric Inductive Bias of Grokking | 2603.05228 | Uniform attention still groks |
| Systematic Empirical Study | 2603.25009 | Weight decay dominates, architecture effects small |

---

## Cold Reader Feedback (Agent Review of Published Blog)

A first-principles-scientist agent read the blog with zero context. Key feedback:

### What Works
- Post is clearly understood. Clean throughline.
- Charts support claims. Bar chart comparing 7K vs 11K is the strongest visual.
- Writing feels authentic, not AI-generated.
- Limitations section is unusually honest.

### What Needs Fixing
1. **Duplicate sentence**: The `<Emphasis>` block after the intro repeats the preceding paragraph almost verbatim. Cut one.
2. **"Entry layer" / "exit layer" undefined**: Never explicitly says entry = L0, exit = L3. Should be stated once.
3. **"Exit MLP" vs "freeze all MLP"** could confuse readers. One is last layer's MLP, other is all layers.
4. **Freeze exit layer from 11K gets 45% (not grokked)** but the post breaks it into exit attention + exit MLP without reconciling that the whole layer still blocks. This is an interesting interaction effect - deserves a sentence.
5. **"Freeze all attention" from 11K gets 39%** - jumped from 17% in the last measurement. Something IS changing. Worth a remark.
6. **"Freeze exit attention" gets 67% from 7K and 65% from 11K** - it gets WORSE, contradicting the infrastructure narrative. Post doesn't engage with this.
7. **Title slightly overstated**: "You can see the boundary" but only two checkpoints = 4,000-step window.
8. **"Infrastructure" is a label, not an explanation.** Used as if it explains something when it's just naming the observation.

### What Would Make It Stronger
1. **Multiple seeds** (biggest gap - everything is one run)
2. **Finer checkpoint sweep** (8K, 9K, 10K) to locate boundary precisely
3. **Mechanistic peek** inside entry layer at both checkpoints
4. **Longer training** per intervention to distinguish "blocked" from "delayed"

---

## Action Items (Priority Order)

### Must Do Before Publishing
1. **Run ablation for longer** (20-30K steps instead of 10K) to resolve the "still rising" issue. This is the biggest data integrity problem. Several results classified as "blocked" may actually be "delayed."
2. **Fix the duplicate Emphasis block** in the intro.
3. **Define entry/exit layer** explicitly (L0, L3) in the setup section.
4. **Add a sentence about freeze-exit-layer paradox** (exit attn + exit MLP grok separately, but whole exit layer doesn't).
5. **Acknowledge the freeze-exit-attn anomaly** (65% from 11K is lower than 67% from 7K).

### Should Do (Strengthens the Paper Significantly)
6. **Run 2-3 additional seeds** with the same ablation protocol.
7. **Finer checkpoint sweep**: test at 8K, 9K, 10K to narrow the phase boundary.
8. **Update charts and blog post** with corrected data after longer runs.

### Nice to Have
9. **Mechanistic analysis**: look at entry layer attention patterns at 7K vs 11K.
10. **Connect to Nanda's Fourier progress measures**: when does the infrastructure phase end in Fourier space?

---

## Technical Setup

### Running the Experiment
```bash
# Step 1: Train base model + save checkpoints (~20 min on MPS/GPU)
python grokking_full_metrics.py

# Step 2: Run ablation study (~3-4 hours, 22 runs x 10K steps each)
python grokking_ablation.py
```

Device: Uses MPS (Apple Silicon) or CUDA. Falls back to CPU.
Model: 4-layer transformer, 128 hidden, 4 heads, 820K params.
Task: Modular division (a/b mod 97), 50% train split.
Training: AdamW, lr=3e-4, weight decay=0.3, batch=64.

### To Extend the Ablation to 30K Steps
Change `num_steps=10000` to `num_steps=30000` in `grokking_ablation.py` line 175 (`run_ablation` function). This will take ~3x longer but resolve the "still rising" ambiguity.

### Blog Repo
The blog lives at `/Users/kumardivyarajat/WebstormProjects/kumardivyarajat/`. Dev server: `npm run dev` then `http://localhost:3000`. The blog post is at `content/blog/grokking-has-two-phases-and-you-can-see-the-boundary.mdx`.

---

## Rules

- **NEVER hallucinate URLs.** Do not construct GitHub links or any URL unless explicitly provided by the user or verified via web search. A fake GitHub URL was accidentally inserted into earlier blog posts and had to be removed.
- **Never use em dashes.** Use hyphens surrounded by spaces: " - "
- **Never share raw experiment files in the blog.** Always create a cleaned copy in `content/blog/code/`. Grep for tokens, keys, internal paths before sharing.
- **Always confirm with the user** before adding code files to the blog.
- **When running experiments on RunPod**, always use `run-remote.sh` from the evolution-theory repo workflow. Source `/etc/environment` for HF_TOKEN. Set TMPDIR=/workspace/tmp.
