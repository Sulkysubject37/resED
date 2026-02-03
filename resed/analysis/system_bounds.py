"""
System Bound Lifting (Phase 10-B).

Formalizes the relationship between component failure modes and RLCS governance.
Expresses system-level guarantees as inequalities derived from empirical envelopes.
"""

from resed.analysis.failure_envelopes import load_envelopes

class SystemBounds:
    def __init__(self):
        self.envelopes = load_envelopes()
        self.bounds = {}
        
    def compute_bounds(self):
        """
        Derive system-level bounds from component envelopes.
        """
        # 1. Encoder Observability Bound
        # Guarantee: Radial Inflation (Noise) <= Inverse_Envelope(RLCS_D)
        # We want to show that if Noise increases, RLCS_D increases.
        # Bound: RLCS_D >= f(Noise)
        if "resENC_Noise_vs_RLCS_D" in self.envelopes:
            env = self.envelopes["resENC_Noise_vs_RLCS_D"]
            # To claim observability, we need monotonic increase.
            # Check monotonicity of the envelope points.
            ys = [p[1] for p in env.points]
            is_monotonic = all(y2 >= y1 for y1, y2 in zip(ys, ys[1:]))
            
            self.bounds["ENC_Observability"] = {
                "metric": "RLCS ResLik Score",
                "stress": "Input Noise",
                "relation": "Monotonic Positive" if is_monotonic else "Non-Monotonic",
                "min_detection_threshold": 3.0, # Standard TAU
                "implied_noise_limit": self._inverse_lookup(env, 3.0)
            }

    def _inverse_lookup(self, envelope, y_target):
        """Find x where y >= y_target."""
        for x, y in envelope.points:
            if y >= y_target:
                return x
        return None

    def report(self):
        """Generate a text report of bounds."""
        lines = ["# Formal System Bounds", ""]
        
        for name, data in self.bounds.items():
            lines.append(f"## {name}")
            for k, v in data.items():
                lines.append(f"* **{k}**: {v}")
            lines.append("")
            
        return "\n".join(lines)

if __name__ == "__main__":
    sb = SystemBounds()
    sb.compute_bounds()
    print(sb.report())
