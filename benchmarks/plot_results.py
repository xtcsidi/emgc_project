import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    csv_path = "simulation_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the simulation first.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Ensure output directory exists
    output_dir = "../report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a highly refined, premium aesthetic
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'grid.color': '#eeeeee',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })

    # Precision mapping
    prec_map = {'int4': 1, 'int8': 2, 'fp16': 3}
    df['precision_num'] = df['precision'].map(prec_map)

    # -------------------------------------------------------------------------
    # Plot 1: VRAM Usage % vs. Precision State
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, dpi=300)
    gpus = ["GTX 1060", "RTX 2060", "RX 580"]
    
    # Professional harmonious palette
    vram_color = '#00a896'  # Sleek teal-cyan
    prec_color = '#d62828'  # Rich crimson-red
    
    for i, gpu in enumerate(gpus):
        ax1 = axes[i]
        gpu_df = df[df['gpu'] == gpu].sort_values('round')
        
        # Primary axis: VRAM %
        vram_pct_pct = gpu_df['vram_pct'] * 100
        line1 = ax1.plot(gpu_df['round'], vram_pct_pct, color=vram_color, linewidth=2.5, 
                         marker='o', markersize=5, label='VRAM Usage %')
        # Fill under curve for a beautiful premium look
        ax1.fill_between(gpu_df['round'], vram_pct_pct, color=vram_color, alpha=0.15)
        
        ax1.set_ylabel('VRAM Usage (%)', color='#028090', fontweight='semibold')
        ax1.tick_params(axis='y', labelcolor='#028090')
        ax1.set_ylim(0, 105)
        ax1.grid(True)
        
        # Add VRAM threshold line at 80%
        ax1.axhline(80, color='#f77f00', linestyle='--', linewidth=1.2, alpha=0.8, 
                    label='Quantization Threshold (80%)')
        
        # Secondary axis: Precision
        ax2 = ax1.twinx()
        line2 = ax2.step(gpu_df['round'], gpu_df['precision_num'], where='mid', 
                         color=prec_color, linewidth=2.2, linestyle='-', marker='s', 
                         markersize=5, label='Precision Level')
        
        ax2.set_ylabel('Precision State', color=prec_color, fontweight='semibold')
        ax2.tick_params(axis='y', labelcolor=prec_color)
        ax2.set_ylim(0.5, 3.5)
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(['INT4', 'INT8', 'FP16'])
        
        # Title & decorations
        ax1.set_title(f"{gpu} Elastic Quantization Behavior", fontsize=13, fontweight='bold', pad=10, color='#2c3e50')
        
        # Combine legends
        if i == 0:
            lines = line1 + [ax1.lines[-1]] + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd', framealpha=0.95)

    axes[-1].set_xlabel('Simulation Round', fontweight='semibold', labelpad=10)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "vram_vs_precision.png")
    plt.savefig(plot1_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Plot 1: {plot1_path}")

    # -------------------------------------------------------------------------
    # Plot 2: Temperature vs Sparsity
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Temperature (common scenario profile)
    # We take GTX 1060's temp since it represents the scenario temperature
    scenario_df = df[df['gpu'] == "GTX 1060"].sort_values('round')
    
    temp_color = '#e76f51'  # Warm terracotta orange
    line_temp = ax1.plot(scenario_df['round'], scenario_df['temp_c'], color=temp_color, 
                         linewidth=3, marker='D', markersize=6, label='GPU Temp (°C)')
    ax1.fill_between(scenario_df['round'], scenario_df['temp_c'], 40, color=temp_color, alpha=0.1)
    
    # Thermal threshold line at 80°C
    ax1.axhline(80, color='#d62828', linestyle='--', linewidth=1.5, alpha=0.8, 
                label='Thermal Pruning Trigger (80°C)')
    
    ax1.set_xlabel('Simulation Round', fontweight='semibold', labelpad=10)
    ax1.set_ylabel('GPU Temperature (°C)', color='#c94a29', fontweight='semibold')
    ax1.tick_params(axis='y', labelcolor='#c94a29')
    ax1.set_ylim(35, 95)
    ax1.grid(True)
    
    # Secondary axis: Sparsity for all 3 GPUs
    ax2 = ax1.twinx()
    
    gpu_colors = {
        'GTX 1060': '#2a9d8f',  # Muted teal
        'RTX 2060': '#264653',  # Deep slate blue
        'RX 580': '#457b9d'     # Steel blue
    }
    
    lines_sparsity = []
    for gpu in gpus:
        gpu_df = df[df['gpu'] == gpu].sort_values('round')
        sparsity_pct = gpu_df['sparsity'] * 100
        line_sp = ax2.plot(gpu_df['round'], sparsity_pct, color=gpu_colors[gpu], 
                           linewidth=2, linestyle=':', marker='^', markersize=5, 
                           label=f'{gpu} Sparsity %')
        lines_sparsity.extend(line_sp)
        
    ax2.set_ylabel('Weight Sparsity (%)', color='#1d3557', fontweight='semibold')
    ax2.tick_params(axis='y', labelcolor='#1d3557')
    ax2.set_ylim(-2, 25)
    
    # Title & Legend
    plt.title("GPU Thermal Profile vs. Post-Pruning Sparsity Dynamics", fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    
    # Combine legends beautifully
    all_lines = line_temp + [ax1.lines[-1]] + lines_sparsity
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "temp_vs_sparsity.png")
    plt.savefig(plot2_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Plot 2: {plot2_path}")

if __name__ == "__main__":
    generate_plots()
