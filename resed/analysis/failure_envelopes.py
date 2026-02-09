"""
Failure Envelope Extraction.

Computes empirical failure envelopes from component stress test logs.
"""

import os
import pandas as pd
import numpy as np

LOG_DIR = "experiments/component_tests"

class FailureEnvelope:
    """
    Represents an empirical bound on component behavior.
    """
    def __init__(self, x_name, y_name, direction='max'):
        self.x_name = x_name
        self.y_name = y_name
        self.direction = direction
        self.points = []

    def fit(self, df):
        if self.direction == 'max':
            grouped = df.groupby(self.x_name)[self.y_name].max()
        else:
            grouped = df.groupby(self.x_name)[self.y_name].min()
            
        self.points = sorted([(x, y) for x, y in grouped.items()])
        
    def get_bound(self, x_query):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return np.interp(x_query, xs, ys)

    def to_dict(self):
        return {
            "x": self.x_name,
            "y": self.y_name,
            "direction": self.direction,
            "points": self.points
        }

def load_envelopes():
    """
    Load logs and compute envelopes for all components.
    """
    envelopes = {}
    
    # 1. resENC
    enc_path = os.path.join(LOG_DIR, "resenc_stability_log.csv")
    if os.path.exists(enc_path):
        df_enc = pd.read_csv(enc_path)
        df_noise = df_enc[df_enc["Perturbation"] == "Noise"]
        
        env_d = FailureEnvelope("Intensity", "RLCS_D", "max")
        env_d.fit(df_noise)
        envelopes["resENC_Noise_vs_RLCS_D"] = env_d
        
        env_cos = FailureEnvelope("Intensity", "Cosine_Sim", "min")
        env_cos.fit(df_noise)
        envelopes["resENC_Noise_vs_Cosine"] = env_cos
        
        env_var = FailureEnvelope("Intensity", "Var_Inflation", "max")
        env_var.fit(df_noise)
        envelopes["resENC_Noise_vs_Var"] = env_var

    # 2. resTR
    tr_path = os.path.join(LOG_DIR, "restr_sensitivity_log.csv")
    if os.path.exists(tr_path):
        df_tr = pd.read_csv(tr_path)
        
        env_ent = FailureEnvelope("Severity", "Attn_Entropy", "min")
        env_ent.fit(df_tr)
        envelopes["resTR_Corruption_vs_Entropy"] = env_ent
        
        env_conc = FailureEnvelope("Severity", "Max_Attn", "max")
        env_conc.fit(df_tr)
        envelopes["resTR_Corruption_vs_Concentration"] = env_conc

    # 3. resDEC
    dec_path = os.path.join(LOG_DIR, "resdec_volatility_log.csv")
    if os.path.exists(dec_path):
        df_dec = pd.read_csv(dec_path)
        
        env_div = FailureEnvelope("Noise", "Output_Divergence", "max")
        env_div.fit(df_dec)
        envelopes["resDEC_Noise_vs_Divergence"] = env_div
        
        env_sens = FailureEnvelope("Noise", "Sensitivity_Ratio", "max")
        env_sens.fit(df_dec)
        envelopes["resDEC_Noise_vs_Sensitivity"] = env_sens

    return envelopes