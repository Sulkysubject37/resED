"""
Universality Validation.

Executes RLCS governance across heterogeneous architectures (resENC, VAE, Transformer)
to demonstrate representation-level invariance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

from experiments.universality.models import WrapperResENC, WrapperVAE, WrapperTransformer
from resed.rlcs.control_surface import rlcs_control
from resed.rlcs.types import RlcsSignal
from resed.calibration.calibrator import RlcsCalibrator

# Config
OUTPUT_DIR = "docs/figures"
D_IN = 128
D_Z = 64
BATCH_SIZE = 200

def perturb_input_noise(x, sigma):
    rng = np.random.default_rng(42)
    return x + rng.normal(0, sigma, x.shape)

def perturb_input_shock(x, prob=0.05, factor=10.0):
    rng = np.random.default_rng(42)
    mask = rng.random(x.shape[0]) < prob
    x_out = x.copy()
    x_out[mask] *= factor
    return x_out

def perturb_input_drift(x, factor=2.0):
    return x + factor

def perturb_input_dropout(x, prob=0.2):
    rng = np.random.default_rng(42)
    mask = rng.random(x.shape) > prob
    return x * mask

def run_experiment():
    print("Starting Universality Validation...")
    
    models = {
        "resENC": WrapperResENC(D_IN, D_Z),
        "VAE": WrapperVAE(D_IN, D_Z),
        "Transformer": WrapperTransformer(D_IN, D_Z)
    }
    
    rng = np.random.default_rng(42)
    X_clean = rng.normal(0, 1, (BATCH_SIZE, D_IN))
    
    scenarios = [
        ("Clean", lambda x: x),
        ("Noise (0.1)", lambda x: perturb_input_noise(x, 0.1)),
        ("Noise (0.5)", lambda x: perturb_input_noise(x, 0.5)),
        ("Shock (5%)", lambda x: perturb_input_shock(x, prob=0.05)),
        ("Drift (+2.0)", lambda x: perturb_input_drift(x, 2.0)),
        ("Dropout (20%)", lambda x: perturb_input_dropout(x, 0.2))
    ]
    
    results = []
    
    for model_name, model in models.items():
        print(f"Testing {model_name}...")
        
        Z_ref = model.encode(X_clean)
        mu_ref = np.mean(Z_ref, axis=0)
        sigma_ref = np.std(Z_ref, axis=0)
        sigma_scalar = np.mean(sigma_ref)
        if sigma_scalar < 1e-6: sigma_scalar = 1.0
        
        s_dummy = np.zeros((BATCH_SIZE, 4))
        diag_ref = {}
        rlcs_control(Z_ref, s_dummy, diagnostics=diag_ref, mu=mu_ref, sigma=sigma_scalar)
        calibrator = RlcsCalibrator()
        calibrator.fit(diag_ref)
        
        for sc_name, pert_func in scenarios:
            X_pert = pert_func(X_clean)
            Z_pert = model.encode(X_pert)
            
            diag = {}
            signals = rlcs_control(Z_pert, s_dummy, diagnostics=diag, calibrator=calibrator, mu=mu_ref, sigma=sigma_scalar)
            
            raw_d = diag['population_consistency']
            cal_d = calibrator.calibrate_batch('population_consistency', raw_d)
            mean_cal_d = np.mean(cal_d)
            
            n_proceed = signals.count(RlcsSignal.PROCEED)
            n_abstain = signals.count(RlcsSignal.ABSTAIN)
            
            results.append({
                "Model": model_name,
                "Scenario": sc_name,
                "Mean_Z_Score": mean_cal_d,
                "Pct_PROCEED": n_proceed / BATCH_SIZE,
                "Pct_ABSTAIN": n_abstain / BATCH_SIZE
            })

    df = pd.DataFrame(results)
    print(df)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    pivot_d = df.pivot(index="Scenario", columns="Model", values="Mean_Z_Score")
    pivot_d = pivot_d.reindex([s[0] for s in scenarios])
    pivot_d.plot(kind='bar', ax=ax1, width=0.8)
    
    ax1.axhline(3.0, color='red', linestyle='--', label='Threshold (3.0)')
    ax1.set_title("Universality: Calibrated Sensor Response across Architectures")
    ax1.set_ylabel("Mean ResLik Z-Score")
    ax1.legend(title="Model")
    ax1.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_universality_sensor_response.pdf"))
    
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    pivot_abs = df.pivot(index="Scenario", columns="Model", values="Pct_ABSTAIN")
    pivot_abs = pivot_abs.reindex([s[0] for s in scenarios])
    pivot_abs.plot(kind='bar', ax=ax2, width=0.8, colormap='viridis')
    
    ax2.set_title("Universality: Governance Consistency (Abstention Rate)")
    ax2.set_ylabel("Fraction ABSTAIN")
    ax2.legend(title="Model")
    ax2.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_universality_governance.pdf"))
    print("Figures saved.")

if __name__ == "__main__":
    run_experiment()