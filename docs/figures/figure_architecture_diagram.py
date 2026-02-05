"""
Generate a high-quality, professional architecture diagram for resED.
Focuses on clarity, component isolation, and professional scientific aesthetics.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    # Use a clean style
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color Palette (Professional/Scientific)
    COLOR_COMPONENT = '#dae8fc'  # Light Blue
    COLOR_GOVERNANCE = '#fff2cc' # Light Yellow
    COLOR_SIGNAL = '#f8cecc'     # Light Red/Pink
    COLOR_EDGE = '#333333'       # Dark Gray

    # Helper function for boxes
    def add_component(x, y, w, h, text, color, header=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     fc=color, ec=COLOR_EDGE, lw=1.5)
        ax.add_patch(rect)
        if header:
            ax.text(x + w/2, y + h + 0.2, header, ha='center', va='bottom', fontweight='bold', fontsize=13)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center')

    # Coordinates
    Y_MAIN = 5.0
    Y_GOV = 1.5
    
    # 1. Main Pipeline
    # Input
    ax.text(1.0, Y_MAIN + 0.5, "Input $x$", ha='center', va='center', fontweight='bold')
    
    # resENC
    add_component(2.5, Y_MAIN, 2.0, 1.0, "resENC\n$z = f(x)$", COLOR_COMPONENT, "Encoder")
    
    # Latent Space
    ax.text(5.75, Y_MAIN + 0.5, "Latent $z$", ha='center', va='center', fontweight='bold', color='#666666')
    
    # resTR
    add_component(7.5, Y_MAIN, 2.0, 1.0, "resTR\n$z_{ref} = z + \Delta z$", COLOR_COMPONENT, "Refinement")
    
    # resDEC
    add_component(11.0, Y_MAIN, 2.0, 1.0, "resDEC\n$y = g(z)$", COLOR_COMPONENT, "Decoder")
    
    # Output
    ax.text(14.5, Y_MAIN + 0.5, "Output $y$", ha='center', va='center', fontweight='bold')

    # 2. RLCS Governance Layer (Below)
    # Governance Box
    gov_rect = patches.FancyBboxPatch((5.0, Y_GOV - 0.5), 8.0, 2.0, boxstyle="round,pad=0.2", 
                                     fc=COLOR_GOVERNANCE, ec='#d6b656', lw=2, linestyle='--')
    ax.add_patch(gov_rect)
    ax.text(5.2, Y_GOV + 1.2, "RLCS Governance Layer", ha='left', va='bottom', fontweight='bold', color='#826a13')

    # Internal components of RLCS
    add_component(5.5, Y_GOV, 1.5, 0.8, "Sensors\n$D, T, A$", '#ffffff')
    add_component(8.0, Y_GOV, 1.5, 0.8, "Calibration\n$Z$-Mapping", '#ffffff')
    
    # Decision Logic (Diamond)
    sig_x, sig_y = 11.5, Y_GOV + 0.4
    diamond = patches.RegularPolygon((sig_x, sig_y), 4, radius=0.7, fc=COLOR_SIGNAL, ec='#b85450', lw=2)
    ax.add_patch(diamond)
    ax.text(sig_x, sig_y, "Signal $\pi$", ha='center', va='center', fontweight='bold')

    # --- Arrows (Data Flow) ---
    def draw_arrow(start, end, style="->", color=COLOR_EDGE, ls='-'):
        ax.annotate("", xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle=style, color=color, lw=2, shrinkA=5, shrinkB=5, ls=ls))

    # Main Path
    draw_arrow((1.5, Y_MAIN + 0.5), (2.5, Y_MAIN + 0.5))   # To Enc
    draw_arrow((4.5, Y_MAIN + 0.5), (5.0, Y_MAIN + 0.5))   # To Latent
    draw_arrow((6.5, Y_MAIN + 0.5), (7.5, Y_MAIN + 0.5))   # To TR
    draw_arrow((9.5, Y_MAIN + 0.5), (11.0, Y_MAIN + 0.5))  # To DEC
    draw_arrow((13.0, Y_MAIN + 0.5), (13.8, Y_MAIN + 0.5)) # To Out

    # Governance Loop
    draw_arrow((5.75, Y_MAIN), (5.75, Y_GOV + 0.8)) # Z to Sensors
    draw_arrow((7.0, Y_GOV + 0.4), (8.0, Y_GOV + 0.4)) # Sensors to Calib
    draw_arrow((9.5, Y_GOV + 0.4), (10.8, Y_GOV + 0.4)) # Calib to Signal

    # Gating Signals (Dashed Red Arrows)
    draw_arrow((sig_x, sig_y + 0.7), (8.5, Y_MAIN), style="-|>", color='#b85450', ls='--') # To TR
    draw_arrow((sig_x, sig_y + 0.7), (12.0, Y_MAIN), style="-|>", color='#b85450', ls='--') # To DEC
    
    plt.savefig("docs/phase11_formal_report/figures/architecture_diagram.pdf", bbox_inches='tight')
    plt.savefig("docs/phase11_formal_report/figures/architecture_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_diagram()
    print("Diagram saved.")