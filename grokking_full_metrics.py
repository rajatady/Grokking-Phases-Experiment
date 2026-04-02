"""
GROKKING EXPERIMENT WITH FULL METRICS

Question: Do r, α predict generalization? Or do we need more metrics?

Additional metrics to track:
- Weight norms per layer (L2)
- Weight effective rank
- Gradient norms
- Attention entropy
- MLP activation sparsity
- Representation similarity (train vs test)
- Loss Hessian trace (curvature)

Run for 50k steps with higher weight decay to trigger grokking.
"""

import argparse
import json
from pathlib import Path
import copy
from collections import defaultdict

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

P = 97
TRAIN_FRAC = 0.5


def mod_inverse(b, p):
    return pow(b, p - 2, p)


class ModularDivisionDataset:
    def __init__(self, p, train=True, train_frac=0.5, seed=42):
        self.p = p
        np.random.seed(seed)
        all_pairs = [(a, b) for a in range(p) for b in range(1, p)]
        np.random.shuffle(all_pairs)
        split = int(len(all_pairs) * train_frac)
        self.pairs = all_pairs[:split] if train else all_pairs[split:]
        self.pairs = [(a, b, (a * mod_inverse(b, p)) % p) for a, b in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.pairs), min(batch_size, len(self.pairs)), replace=False)
        batch = [self.pairs[i] for i in indices]
        x = torch.tensor([[a, b, P] for a, b, c in batch], dtype=torch.long)
        y = torch.tensor([c for a, b, c in batch], dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)


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

    def forward(self, x, return_intermediates=False):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))

        intermediates = {'hidden_states': [h.clone()], 'attn_weights': [], 'mlp_activations': []}

        for layer in self.layers:
            h, attn_w, mlp_act = layer(h, return_extras=True)
            intermediates['hidden_states'].append(h.clone())
            intermediates['attn_weights'].append(attn_w)
            intermediates['mlp_activations'].append(mlp_act)

        h = self.ln_f(h)
        logits = self.head(h[:, -1])

        if return_intermediates:
            return logits, intermediates
        return logits


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp_fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.mlp_fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x, return_extras=False):
        h = self.ln1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True)
        x = x + attn_out

        h = self.ln2(x)
        mlp_hidden = F.gelu(self.mlp_fc1(h))
        mlp_out = self.mlp_fc2(mlp_hidden)
        x = x + mlp_out

        if return_extras:
            return x, attn_weights, mlp_hidden
        return x


def compute_full_metrics(model, train_data, test_data, batch_size=128):
    """Compute ALL metrics we can think of."""
    model.eval()

    metrics = {}

    # Get batches
    x_train, y_train = train_data.get_batch(batch_size)
    x_test, y_test = test_data.get_batch(batch_size)

    with torch.no_grad():
        # Forward pass with intermediates
        logits_train, inter_train = model(x_train, return_intermediates=True)
        logits_test, inter_test = model(x_test, return_intermediates=True)

        # === 1. Basic accuracy/loss ===
        train_acc = (logits_train.argmax(-1) == y_train).float().mean().item()
        test_acc = (logits_test.argmax(-1) == y_test).float().mean().item()
        train_loss = F.cross_entropy(logits_train, y_train).item()
        test_loss = F.cross_entropy(logits_test, y_test).item()

        metrics['train_acc'] = train_acc
        metrics['test_acc'] = test_acc
        metrics['train_loss'] = train_loss
        metrics['test_loss'] = test_loss
        metrics['generalization_gap'] = train_acc - test_acc

        # === 2. Per-layer r, α, direction preservation ===
        metrics['per_layer'] = {}
        hidden_train = inter_train['hidden_states']

        for l in range(model.num_layers):
            h_in = hidden_train[l][:, -1].mean(0).cpu().float()
            h_out = hidden_train[l + 1][:, -1].mean(0).cpu().float()
            delta = h_out - h_in

            h_in_norm = h_in.norm().item()
            r = delta.norm().item() / h_in_norm if h_in_norm > 1e-8 else 0
            alpha = F.cosine_similarity(h_in.unsqueeze(0), delta.unsqueeze(0)).item() if h_in_norm > 1e-8 else 0
            dir_pres = F.cosine_similarity(h_in.unsqueeze(0), h_out.unsqueeze(0)).item() if h_in_norm > 1e-8 else 0

            metrics['per_layer'][l] = {
                'r': r,
                'alpha': alpha,
                'dir_pres': dir_pres,
            }

        # === 3. Weight norms per layer ===
        metrics['weight_norms'] = {}
        total_weight_norm = 0
        for l, layer in enumerate(model.layers):
            mlp_norm = sum(p.norm().item()**2 for p in layer.mlp_fc1.parameters()) + \
                       sum(p.norm().item()**2 for p in layer.mlp_fc2.parameters())
            mlp_norm = mlp_norm ** 0.5

            attn_norm = sum(p.norm().item()**2 for p in layer.attn.parameters()) ** 0.5

            metrics['weight_norms'][l] = {
                'mlp': mlp_norm,
                'attn': attn_norm,
                'total': (mlp_norm**2 + attn_norm**2) ** 0.5
            }
            total_weight_norm += mlp_norm**2 + attn_norm**2

        metrics['total_weight_norm'] = total_weight_norm ** 0.5

        # === 4. Attention entropy (how focused is attention?) ===
        metrics['attention_entropy'] = {}
        for l, attn_w in enumerate(inter_train['attn_weights']):
            # attn_w shape: [batch, heads, seq, seq]
            # Compute entropy of attention distribution
            attn_w = attn_w.mean(0).mean(0)  # Average over batch and heads -> [seq, seq]
            attn_w = attn_w[-1]  # Last position attending to all -> [seq]
            attn_w = attn_w + 1e-10
            entropy = -(attn_w * torch.log(attn_w)).sum().item()
            metrics['attention_entropy'][l] = entropy

        # === 5. MLP activation sparsity ===
        metrics['mlp_sparsity'] = {}
        for l, mlp_act in enumerate(inter_train['mlp_activations']):
            # Fraction of near-zero activations
            sparsity = (mlp_act.abs() < 0.01).float().mean().item()
            # Also compute mean activation magnitude
            mean_act = mlp_act.abs().mean().item()
            metrics['mlp_sparsity'][l] = {
                'sparsity': sparsity,
                'mean_activation': mean_act,
            }

        # === 6. Representation similarity (train vs test at each layer) ===
        metrics['repr_similarity'] = {}
        hidden_test = inter_test['hidden_states']
        for l in range(model.num_layers + 1):
            h_train_l = hidden_train[l][:, -1].mean(0)  # [hidden]
            h_test_l = hidden_test[l][:, -1].mean(0)
            sim = F.cosine_similarity(h_train_l.unsqueeze(0), h_test_l.unsqueeze(0)).item()
            metrics['repr_similarity'][l] = sim

        # === 7. Effective rank of hidden representations ===
        metrics['effective_rank'] = {}
        for l in range(model.num_layers + 1):
            h = hidden_train[l][:, -1]  # [batch, hidden]
            # Compute SVD
            try:
                U, S, V = torch.svd(h)
                # Effective rank = exp(entropy of normalized singular values)
                S_norm = S / S.sum()
                S_norm = S_norm + 1e-10
                eff_rank = torch.exp(-(S_norm * torch.log(S_norm)).sum()).item()
                metrics['effective_rank'][l] = eff_rank
            except:
                metrics['effective_rank'][l] = 0

        # === 8. Output entropy ===
        probs = F.softmax(logits_train, dim=-1)
        output_entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
        metrics['output_entropy'] = output_entropy

    # === 9. Gradient norm (requires backward) ===
    model.train()
    x, y = train_data.get_batch(64)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    total_grad_norm = 0
    metrics['grad_norms'] = {}
    for l, layer in enumerate(model.layers):
        layer_grad_norm = 0
        for p in layer.parameters():
            if p.grad is not None:
                layer_grad_norm += p.grad.norm().item() ** 2
        layer_grad_norm = layer_grad_norm ** 0.5
        metrics['grad_norms'][l] = layer_grad_norm
        total_grad_norm += layer_grad_norm ** 2

    metrics['total_grad_norm'] = total_grad_norm ** 0.5

    # Zero gradients
    model.zero_grad()
    model.eval()

    return metrics


def run_grokking_experiment(name, model, train_data, test_data, num_steps=50000,
                            checkpoint_every=500, lr=1e-3, weight_decay=1.0):
    """Run experiment tracking ALL metrics."""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"Steps: {num_steps}, LR: {lr}, Weight Decay: {weight_decay}")
    print("="*70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = defaultdict(list)
    checkpoints = {}

    for step in range(num_steps + 1):
        if step % checkpoint_every == 0:
            metrics = compute_full_metrics(model, train_data, test_data)

            # Store everything
            history['step'].append(step)
            history['train_acc'].append(metrics['train_acc'])
            history['test_acc'].append(metrics['test_acc'])
            history['train_loss'].append(metrics['train_loss'])
            history['test_loss'].append(metrics['test_loss'])
            history['gap'].append(metrics['generalization_gap'])
            history['total_weight_norm'].append(metrics['total_weight_norm'])
            history['total_grad_norm'].append(metrics['total_grad_norm'])
            history['output_entropy'].append(metrics['output_entropy'])

            # Per-layer metrics
            for l in range(model.num_layers):
                history[f'L{l}_r'].append(metrics['per_layer'][l]['r'])
                history[f'L{l}_alpha'].append(metrics['per_layer'][l]['alpha'])
                history[f'L{l}_dir_pres'].append(metrics['per_layer'][l]['dir_pres'])
                history[f'L{l}_weight_norm'].append(metrics['weight_norms'][l]['total'])
                history[f'L{l}_attn_entropy'].append(metrics['attention_entropy'][l])
                history[f'L{l}_mlp_sparsity'].append(metrics['mlp_sparsity'][l]['sparsity'])
                history[f'L{l}_repr_sim'].append(metrics['repr_similarity'][l])
                history[f'L{l}_eff_rank'].append(metrics['effective_rank'][l])
                history[f'L{l}_grad_norm'].append(metrics['grad_norms'][l])

            # Region summaries
            exit_l = model.num_layers - 1
            mid_l = model.num_layers // 2

            if step % 2000 == 0:
                print(f"Step {step:>6}: loss={metrics['train_loss']:.4f}, "
                      f"train={metrics['train_acc']:.3f}, test={metrics['test_acc']:.3f}, "
                      f"gap={metrics['generalization_gap']:.3f} | "
                      f"exit_r={metrics['per_layer'][exit_l]['r']:.3f}, "
                      f"W_norm={metrics['total_weight_norm']:.1f}")

            # Save checkpoints at the same cadence as metric collection so ablations
            # can start from intermediate plateaus like 7K and 11K.
            if step % checkpoint_every == 0:
                checkpoints[step] = {
                    'model_state': copy.deepcopy(model.state_dict()),
                    'metrics': metrics,
                }

        if step < num_steps:
            model.train()
            x, y = train_data.get_batch(64)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return dict(history), checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Train the grokking baseline and save checkpoints/metrics.")
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", default="grokking_full_metrics.json")
    parser.add_argument("--checkpoints-output", default="grokking_checkpoints.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*80)
    print("GROKKING EXPERIMENT WITH FULL METRICS")
    print("="*80)
    print(f"\nTask: a / b mod {P}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {args.seed}")

    train_data = ModularDivisionDataset(P, train=True, seed=args.seed)
    test_data = ModularDivisionDataset(P, train=False, seed=args.seed)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Use smaller model (4 layers) and high weight decay for faster grokking
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SimpleTransformer(num_layers=4).to(DEVICE)

    history, checkpoints = run_grokking_experiment(
        "Grokking (WD=0.3)",
        model, train_data, test_data,
        num_steps=args.num_steps,
        weight_decay=0.3,  # Moderate weight decay - more stable
        lr=3e-4,  # Lower learning rate for stability
    )

    # Analyze: When does grokking happen?
    print("\n" + "="*80)
    print("GROKKING ANALYSIS")
    print("="*80)

    # Find grokking point (test acc > 0.5)
    grok_step = None
    for i, (step, test_acc) in enumerate(zip(history['step'], history['test_acc'])):
        if test_acc > 0.5:
            grok_step = step
            break

    if grok_step:
        print(f"\n✓ GROKKING occurred at step {grok_step}!")

        # What changed at grokking?
        grok_idx = history['step'].index(grok_step)
        pre_grok_idx = max(0, grok_idx - 2)

        print(f"\nMetrics BEFORE grokking (step {history['step'][pre_grok_idx]}):")
        print(f"  test_acc: {history['test_acc'][pre_grok_idx]:.3f}")
        print(f"  L3_r: {history['L3_r'][pre_grok_idx]:.3f}")
        print(f"  L3_alpha: {history['L3_alpha'][pre_grok_idx]:.3f}")
        print(f"  weight_norm: {history['total_weight_norm'][pre_grok_idx]:.1f}")
        print(f"  L3_eff_rank: {history['L3_eff_rank'][pre_grok_idx]:.1f}")

        print(f"\nMetrics AFTER grokking (step {grok_step}):")
        print(f"  test_acc: {history['test_acc'][grok_idx]:.3f}")
        print(f"  L3_r: {history['L3_r'][grok_idx]:.3f}")
        print(f"  L3_alpha: {history['L3_alpha'][grok_idx]:.3f}")
        print(f"  weight_norm: {history['total_weight_norm'][grok_idx]:.1f}")
        print(f"  L3_eff_rank: {history['L3_eff_rank'][grok_idx]:.1f}")
    else:
        print(f"\n✗ No grokking detected (max test_acc = {max(history['test_acc']):.3f})")

    # What correlates with test accuracy?
    print("\n" + "="*80)
    print("CORRELATION WITH TEST ACCURACY")
    print("="*80)

    test_acc = np.array(history['test_acc'])

    correlations = {}
    for key in history:
        if key not in ['step', 'test_acc'] and len(history[key]) == len(test_acc):
            try:
                corr = np.corrcoef(test_acc, history[key])[0, 1]
                if not np.isnan(corr):
                    correlations[key] = corr
            except:
                pass

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Metric':<25} {'Correlation':>12}")
    print("-" * 40)
    for key, corr in sorted_corr[:20]:
        print(f"{key:<25} {corr:>+12.3f}")

    # Save
    results = {
        'seed': args.seed,
        'history': history,
        'grok_step': grok_step,
        'correlations': correlations,
    }

    with open(args.metrics_output, 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(checkpoints, args.checkpoints_output)

    print(f"\nSaved: {args.metrics_output}, {args.checkpoints_output}")


if __name__ == "__main__":
    main()
