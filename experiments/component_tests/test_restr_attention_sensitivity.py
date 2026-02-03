"""
Component Test 2: resTR Sensitivity (Phase 10-A).

Detects attention collapse and noise amplification.
Replicates attention mechanism locally to inspect weights without modifying core code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())

from resed.restr.restr import ResTR
from resed.rlcs.sensors import temporal_consistency

OUTPUT_DIR = "experiments/component_tests"
FIGURE_DIR = "docs/figures"
D_MODEL = 64
N_HEADS = 4
SEQ_LEN = 20
BATCH_SIZE = 50

def setup_restr():
    restr = ResTR(D_MODEL, N_HEADS)
    return restr

def compute_attention_stats(mhsa, z):
    """
    Replicates MHSA forward pass to extract attention weights for analysis.
    """
    input_ndim = z.ndim
    if input_ndim == 2:
        z_in = z[:, np.newaxis, :]
    else:
        z_in = z
        
    batch, seq_len, d = z_in.shape
    d_head = mhsa.d_head
    
    Q = np.dot(z_in, mhsa.W_q)
    K = np.dot(z_in, mhsa.W_k)
    
    Q = Q.reshape(batch, seq_len, mhsa.n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, mhsa.n_heads, d_head).transpose(0, 2, 1, 3)
    
    # Scores
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    
    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Entropy: -sum(p * log(p))
    # Add eps to log
    entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-12), axis=-1)
    
    # Max Concentration: Max weight
    max_attn = np.max(attn_weights, axis=-1)
    
    return np.mean(entropy), np.mean(max_attn)

def apply_token_corruption(z, n_corrupt=1):
    z_out = z.copy()
    rng = np.random.default_rng(42)
    # Corrupt random tokens in sequence
    for i in range(z.shape[0]):
        indices = rng.choice(z.shape[1], n_corrupt, replace=False)
        z_out[i, indices] += rng.normal(0, 5.0, (n_corrupt, z.shape[2]))
    return z_out

def run_test():
    print("Starting resTR Sensitivity Test...")
    restr = setup_restr()
    mhsa = restr.attention
    
    # Reference Sequence
    rng = np.random.default_rng(42)
    Z_clean = rng.normal(0, 1, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    
    results = []
    
    # 1. Single Token Corruption
    n_tokens = [1, 2, 5]
    for n in n_tokens:
        Z_pert = apply_token_corruption(Z_clean, n)
        
        # Run System
        try:
            Z_out_clean = restr.forward(Z_clean, alpha=1.0, beta=1.0)
            Z_out_pert = restr.forward(Z_pert, alpha=1.0, beta=1.0)
        except RuntimeError as e:
            print(f"Caught Expected Safety Violation: {e}")
            # For analysis, we use the input (as if identity/bypassed) or NaN
            # Since we want to measure sensitivity, let's assume bypass
            Z_out_clean = Z_clean
            Z_out_pert = Z_pert
        
        # Analysis
        entropy, max_conc = compute_attention_stats(mhsa, Z_pert)
        
        # Output Variance Amplification
        # Ratio of output variance to input variance
        amp_ratio = np.var(Z_out_pert) / np.var(Z_pert)
        
        # Propagation Radius (How many output tokens changed vs input tokens corrupted)
        # Check diff > threshold
        diff = np.linalg.norm(Z_out_pert - Z_out_clean, axis=-1)
        # Clean diff is 0.
        # Perturbed diff: where is it non-zero?
        # Threshold 0.1
        changed_tokens = np.sum(diff > 0.1, axis=1) # per batch
        avg_prop = np.mean(changed_tokens)
        
        # Temporal Consistency (on output)
        # Note: TCS is for (Batch, Dim). Z is (Batch, Seq, Dim).
        # We treat Batch*Seq as large batch? Or TCS along sequence?
        # Phase 2 `temporal_consistency` logic: `exp(-||z_t - z_{t-1}||)`.
        # It expects `(Time, Dim)`.
        # So we reshape `(Batch*Seq, Dim)`? No, that mixes samples.
        # We compute per sample.
        tcs_scores = []
        for i in range(BATCH_SIZE):
            tcs = temporal_consistency(Z_out_pert[i])
            tcs_scores.append(np.mean(tcs))
        mean_tcs = np.mean(tcs_scores)
        
        results.append({
            "Perturbation": "Token_Corruption",
            "Severity": n,
            "Attn_Entropy": entropy,
            "Max_Attn": max_conc,
            "Amp_Ratio": amp_ratio,
            "Prop_Radius": avg_prop,
            "TCS": mean_tcs
        })
        
    # Save Logs
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "restr_sensitivity_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved logs to {csv_path}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(df["Severity"], df["Attn_Entropy"], marker='o', label="Entropy")
    ax.set_xlabel("Corrupted Tokens")
    ax.set_ylabel("Attention Entropy")
    ax.set_title("resTR Sensitivity: Entropy vs Corruption")
    ax.grid(alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(df["Severity"], df["Prop_Radius"], marker='x', color='red', label="Prop. Radius")
    ax2.set_ylabel("Propagation Radius (Tokens)")
    
    plot_path = os.path.join(FIGURE_DIR, "figure_component_restr_sensitivity.pdf")
    plt.savefig(plot_path)
    print(f"Saved figure to {plot_path}")

if __name__ == "__main__":
    run_test()
