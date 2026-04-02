"""
ABLATION STUDY: What minimal intervention affects generalization?

We load from checkpoint and train with different interventions,
tracking how test accuracy evolves.

Key question: Which interventions affect the test accuracy trajectory?
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = get_device()
print(f"Using device: {DEVICE}", flush=True)
P = 97


def mod_inverse(b, p):
    return pow(b, p - 2, p)


class ModularDivisionDataset:
    def __init__(self, p, train=True, seed=42):
        self.p = p
        np.random.seed(seed)
        all_pairs = [(a, b) for a in range(p) for b in range(1, p)]
        np.random.shuffle(all_pairs)
        split = int(len(all_pairs) * 0.5)
        self.pairs = all_pairs[:split] if train else all_pairs[split:]
        self.pairs = [(a, b, (a * mod_inverse(b, p)) % p) for a, b in self.pairs]

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.pairs), min(batch_size, len(self.pairs)), replace=False)
        batch = [self.pairs[i] for i in indices]
        x = torch.tensor([[a, b, P] for a, b, c in batch], dtype=torch.long)
        y = torch.tensor([c for a, b, c in batch], dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp_fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.mlp_fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(h)))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=99, hidden_size=128, num_layers=4, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h[:, -1])


def apply_intervention(model, intervention_name):
    """Apply a specific intervention."""

    if intervention_name == 'baseline':
        pass

    elif intervention_name == 'freeze_head':
        for p in model.head.parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_embed':
        for p in model.embed.parameters():
            p.requires_grad = False
        for p in model.pos_embed.parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_attn_all':
        for layer in model.layers:
            for p in layer.attn.parameters():
                p.requires_grad = False
            for p in layer.ln1.parameters():
                p.requires_grad = False

    elif intervention_name == 'freeze_mlp_all':
        for layer in model.layers:
            for p in layer.mlp_fc1.parameters():
                p.requires_grad = False
            for p in layer.mlp_fc2.parameters():
                p.requires_grad = False
            for p in layer.ln2.parameters():
                p.requires_grad = False

    elif intervention_name == 'freeze_exit_layer':
        for p in model.layers[-1].parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_entry_layer':
        for p in model.layers[0].parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_middle_layers':
        for p in model.layers[1].parameters():
            p.requires_grad = False
        for p in model.layers[2].parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_exit_attn':
        for p in model.layers[-1].attn.parameters():
            p.requires_grad = False

    elif intervention_name == 'freeze_exit_mlp':
        for p in model.layers[-1].mlp_fc1.parameters():
            p.requires_grad = False
        for p in model.layers[-1].mlp_fc2.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total


def classify_trend(trajectory):
    """Classify the trend: IMPROVING, PLATEAU, or FLAT."""
    if len(trajectory) < 2:
        return 'UNKNOWN', 0

    initial = trajectory[0]['test_acc']
    final = trajectory[-1]['test_acc']
    max_acc = max(t['test_acc'] for t in trajectory)

    # Look at last few points to detect plateau
    last_few = [t['test_acc'] for t in trajectory[-3:]]
    variance = np.var(last_few) if len(last_few) >= 2 else 0

    improvement = final - initial

    if improvement > 0.3:
        trend = 'IMPROVING'
    elif improvement > 0.05:
        trend = 'SLOW_RISE'
    elif abs(improvement) < 0.02:
        trend = 'FLAT'
    else:
        trend = 'OSCILLATING'

    return trend, improvement


def parse_int_list(raw_value):
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_str_list(raw_value):
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def save_results(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def run_ablation(checkpoint_step, intervention_name, checkpoint_path, seed, num_steps=10000, wd=0.3):
    """Run training with a specific intervention."""

    # Load from consolidated checkpoint file (saved by grokking_full_metrics.py)
    all_ckpts = torch.load(checkpoint_path, map_location=DEVICE)
    if checkpoint_step not in all_ckpts:
        raise ValueError(f"Checkpoint {checkpoint_step} not found. Available: {sorted(all_ckpts.keys())}")
    ckpt = all_ckpts[checkpoint_step]

    model = SimpleTransformer(num_layers=4).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])

    trainable, total = apply_intervention(model, intervention_name)

    # Special case: no weight decay
    actual_wd = 0.0 if intervention_name == 'no_weight_decay' else wd

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=actual_wd
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_data = ModularDivisionDataset(P, train=True, seed=seed)
    test_data = ModularDivisionDataset(P, train=False, seed=seed)
    x_test, y_test = test_data.get_batch(256)

    trajectory = []

    print(f"    [trainable: {trainable:,}/{total:,}] Training {num_steps} steps...", flush=True)

    for step in range(num_steps):
        global_step = checkpoint_step + step

        model.train()
        x, y = train_data.get_batch(64)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
                train_logits = model(x)
                train_acc = (train_logits.argmax(-1) == y).float().mean().item()

            trajectory.append({
                'step': global_step,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'loss': loss.item()
            })

            if step % 2000 == 0:
                print(f"      Step {global_step}: train={train_acc:.3f}, test={test_acc:.3f}", flush=True)

    trend, improvement = classify_trend(trajectory)
    initial_acc = trajectory[0]['test_acc'] if trajectory else 0
    final_acc = trajectory[-1]['test_acc'] if trajectory else 0
    max_acc = max(t['test_acc'] for t in trajectory) if trajectory else 0

    print(f"    → {initial_acc:.3f} → {final_acc:.3f} (Δ={improvement:+.3f}) [{trend}]", flush=True)

    return {
        'seed': seed,
        'intervention': intervention_name,
        'checkpoint': checkpoint_step,
        'trainable_params': trainable,
        'total_params': total,
        'initial_test_acc': initial_acc,
        'final_test_acc': final_acc,
        'max_test_acc': max_acc,
        'improvement': improvement,
        'trend': trend,
        'trajectory': trajectory
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run grokking component ablations from saved checkpoints.")
    parser.add_argument("--checkpoint-path", default="grokking_checkpoints.pt")
    parser.add_argument("--output", default="ablation_results.json")
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.3)
    parser.add_argument(
        "--checkpoints",
        default="7000,11000",
        help="Comma-separated checkpoint steps to evaluate, for example 7000,8000,9000,10000,11000",
    )
    parser.add_argument(
        "--interventions",
        default="baseline,no_weight_decay,freeze_head,freeze_embed,freeze_exit_layer,freeze_exit_attn,freeze_exit_mlp,freeze_entry_layer,freeze_middle_layers,freeze_attn_all,freeze_mlp_all",
        help="Comma-separated intervention names to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("ABLATION STUDY: Finding minimal intervention that affects learning")
    print("=" * 70)
    print(f"Seed: {args.seed}")

    checkpoints = parse_int_list(args.checkpoints)
    interventions = parse_str_list(args.interventions)

    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Resuming from {args.output} with {len(all_results)} completed runs")
    else:
        all_results = []

    completed = {(row['checkpoint'], row['intervention']) for row in all_results}

    for ckpt in checkpoints:
        print(f"\n{'='*70}")
        print(f"STARTING FROM CHECKPOINT: Step {ckpt}")
        print("=" * 70)

        for intervention in interventions:
            if (ckpt, intervention) in completed:
                print(f"\n  {intervention}: already completed, skipping", flush=True)
                continue
            print(f"\n  {intervention}:", flush=True)
            result = run_ablation(
                ckpt,
                intervention,
                checkpoint_path=args.checkpoint_path,
                seed=args.seed,
                num_steps=args.num_steps,
                wd=args.weight_decay,
            )
            all_results.append(result)
            save_results(args.output, all_results)

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    print(f"\n{'Intervention':<22} {'Ckpt 7000':<30} {'Ckpt 11000':<30}")
    print(f"{'':22} {'init→final (Δ) [trend]':<30} {'init→final (Δ) [trend]':<30}")
    print("-" * 82)

    for intervention in interventions:
        r7k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 7000]
        r11k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 11000]

        if r7k:
            r7 = r7k[0]
            col7k = f"{r7['initial_test_acc']:.2f}→{r7['final_test_acc']:.2f} ({r7['improvement']:+.2f}) [{r7['trend'][:4]}]"
        else:
            col7k = "N/A"

        if r11k:
            r11 = r11k[0]
            col11k = f"{r11['initial_test_acc']:.2f}→{r11['final_test_acc']:.2f} ({r11['improvement']:+.2f}) [{r11['trend'][:4]}]"
        else:
            col11k = "N/A"

        print(f"{intervention:<22} {col7k:<30} {col11k:<30}")

    # Baseline comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON: Step 7000 vs Step 11000")
    print("=" * 70)

    baseline_7k = [r for r in all_results if r['intervention'] == 'baseline' and r['checkpoint'] == 7000][0]
    baseline_11k = [r for r in all_results if r['intervention'] == 'baseline' and r['checkpoint'] == 11000][0]

    print(f"\nFrom step 7000:  {baseline_7k['initial_test_acc']:.3f} → {baseline_7k['final_test_acc']:.3f} (improvement: {baseline_7k['improvement']:+.3f})")
    print(f"From step 11000: {baseline_11k['initial_test_acc']:.3f} → {baseline_11k['final_test_acc']:.3f} (improvement: {baseline_11k['improvement']:+.3f})")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)

    print("\nInterventions that BLOCK learning (flat trajectory) from BOTH checkpoints:")
    for intervention in interventions:
        r7k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 7000]
        r11k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 11000]
        if r7k and r11k:
            if r7k[0]['trend'] == 'FLAT' and r11k[0]['trend'] == 'FLAT':
                print(f"  - {intervention}")

    print("\nInterventions that BLOCK from 7000 but ALLOW from 11000:")
    for intervention in interventions:
        r7k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 7000]
        r11k = [r for r in all_results if r['intervention'] == intervention and r['checkpoint'] == 11000]
        if r7k and r11k:
            if r7k[0]['improvement'] < 0.05 and r11k[0]['improvement'] > 0.1:
                print(f"  - {intervention}: 7k={r7k[0]['final_test_acc']:.2f}, 11k={r11k[0]['final_test_acc']:.2f}")

    # Save results
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
