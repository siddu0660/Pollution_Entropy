#!/usr/bin/env python3
"""
Generate plots comparing baseline and polluted entropy metrics from JSON data.
Creates a multi-page PDF with plots for all parameters.
"""

import json
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def load_json_data(json_path):
    """Load and parse the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_metric_data(data, metric_name):
    """Extract percentage difference between polluted and baseline values for a specific metric."""
    percentage_differences = []
    epoch_indices = []
    
    # Check if data is a dict (Entropy_Stream.py format) or list (Entropy_EpochClf.py format)
    if isinstance(data, dict) and 'epochs' in data:
        # Entropy_Stream.py format: has metadata, baseline, and epochs
        for epoch_data in data['epochs']:
            epoch_indices.append(epoch_data['epoch_index'])
            baseline = epoch_data['clean_metrics'].get(metric_name, 0)
            polluted = epoch_data['malicious_metrics'].get(metric_name, 0)
            
            # Calculate percentage difference: ((polluted - baseline) / baseline) * 100
            if baseline != 0:
                pct_diff = ((polluted - baseline) / baseline) * 100
            else:
                pct_diff = 0 if polluted == 0 else 100
            percentage_differences.append(pct_diff)
            
    elif isinstance(data, list):
        # Entropy_EpochClf.py format: list of epochs with baseline/polluted
        for epoch_data in data:
            epoch_indices.append(epoch_data['epoch_index'])
            baseline = epoch_data['baseline'].get(metric_name, 0)
            polluted = epoch_data['polluted'].get(metric_name, 0)
            
            # Calculate percentage difference: ((polluted - baseline) / baseline) * 100
            if baseline != 0:
                pct_diff = ((polluted - baseline) / baseline) * 100
            else:
                pct_diff = 0 if polluted == 0 else 100
            percentage_differences.append(pct_diff)
    else:
        raise ValueError("Unknown JSON format. Expected either a list of epochs or a dict with 'epochs' key.")
    
    return epoch_indices, percentage_differences

def group_epochs(epoch_indices, percentage_differences, group_size=10):
    """Group epochs into sets of specified size and average the percentage differences."""
    num_groups = (len(epoch_indices) + group_size - 1) // group_size
    
    grouped_indices = []
    grouped_percentages = []
    
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, len(epoch_indices))
        
        # Use the epoch range as the group label
        grouped_indices.append(f"{epoch_indices[start_idx]}-{epoch_indices[end_idx-1]}")
        
        # Average the percentage differences in this group
        grouped_percentages.append(np.mean(percentage_differences[start_idx:end_idx]))
    
    return grouped_indices, grouped_percentages

def create_percentage_plot(ax, x_labels, percentage_values, metric_name):
    """Create a clean line plot showing percentage differences from baseline."""
    x_pos = np.arange(len(x_labels))
    
    # Separate positive and negative values for different colors
    positive_mask = np.array(percentage_values) >= 0
    
    # Plot the main line
    ax.plot(x_pos, percentage_values, color='#2E4057', linewidth=2.5, 
            marker='o', markersize=6, markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor='#2E4057', alpha=0.9)
    
    # Color the markers based on positive/negative
    for i, (x, y) in enumerate(zip(x_pos, percentage_values)):
        color = '#E63946' if y > 0 else '#06A77D'
        ax.plot(x, y, 'o', markersize=8, color=color, alpha=0.85, zorder=3)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Customize plot
    ax.set_xlabel('Epoch Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - Percentage Difference from Baseline', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # Clean grid
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.15, linestyle='--', linewidth=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', 
               markersize=8, label='Increase from Baseline', alpha=0.85),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#06A77D', 
               markersize=8, label='Decrease from Baseline', alpha=0.85)
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

def generate_plots(json_path, output_pdf=None):
    """Generate all plots and save to PDF."""
    # Load data
    data = load_json_data(json_path)
    
    if not data:
        print("Error: No data found in JSON file")
        return
    
    # Determine output PDF name
    if output_pdf is None:
        # Use the same name as JSON file, just change extension to .pdf
        json_basename = os.path.splitext(json_path)[0]
        output_pdf = f"{json_basename}.pdf"
    
    # Determine epoch count and malicious percentage based on format
    if isinstance(data, dict) and 'epochs' in data:
        # Entropy_Stream.py format
        epoch_count = len(data['epochs'])
        malicious_pct = data.get('metadata', {}).get('malicious_percentage', 'N/A')
    elif isinstance(data, list):
        # Entropy_EpochClf.py format
        epoch_count = len(data)
        malicious_pct = data[0].get('malicious_percent', 'N/A') if data else 'N/A'
    else:
        print("Error: Unknown JSON format")
        return
    
    # Define metrics to plot
    metrics = [
        ('src_ip_entropy', 'Source IP Entropy', 'Entropy (bits)'),
        ('dst_ip_entropy', 'Destination IP Entropy', 'Entropy (bits)'),
        ('src_port_entropy', 'Source Port Entropy', 'Entropy (bits)'),
        ('dst_port_entropy', 'Destination Port Entropy', 'Entropy (bits)'),
        ('src_ip_cardinality', 'Source IP Cardinality', 'Cardinality'),
        ('dst_ip_cardinality', 'Destination IP Cardinality', 'Cardinality'),
        ('src_port_cardinality', 'Source Port Cardinality', 'Cardinality'),
        ('dst_port_cardinality', 'Destination Port Cardinality', 'Cardinality'),
        ('src_ip_uniformity', 'Source IP Uniformity', 'Uniformity'),
        ('dst_ip_uniformity', 'Destination IP Uniformity', 'Uniformity'),
        ('src_dst_ratio', 'Source/Destination Ratio', 'Ratio'),
        ('total_flows', 'Total Flows', 'Flow Count'),
        ('unique_src_ips', 'Unique Source IPs', 'Count'),
        ('unique_dst_ips', 'Unique Destination IPs', 'Count'),
        ('unique_src_ports', 'Unique Source Ports', 'Count'),
        ('unique_dst_ports', 'Unique Destination Ports', 'Count'),
    ]
    
    print(f"\nGenerating plots from: {json_path}")
    print(f"Total epochs: {epoch_count}")
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        for metric_key, metric_name, ylabel in metrics:
            print(f"  Plotting: {metric_name}")
            
            # Extract percentage differences
            epoch_indices, percentage_differences = extract_metric_data(data, metric_key)
            
            # Group epochs into sets of 10
            grouped_indices, grouped_percentages = group_epochs(
                epoch_indices, percentage_differences, group_size=10
            )
            
            # Create figure with single plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Percentage difference plot
            create_percentage_plot(ax, grouped_indices, grouped_percentages, metric_name)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Add metadata page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Entropy Analysis Report', 
                ha='center', fontsize=20, fontweight='bold')
        
        # Summary information
        summary_text = f"""
Data Source: {os.path.basename(json_path)}
Total Epochs Analyzed: {epoch_count}
Epoch Groups (size 10): {len(grouped_indices)}
Malicious Percentage: {malicious_pct}%

Metrics Analyzed:
  • Source IP Entropy
  • Destination IP Entropy
  • Source Port Entropy
  • Destination Port Entropy
  • Source IP Cardinality
  • Destination IP Cardinality
  • Source Port Cardinality
  • Destination Port Cardinality
  • Source IP Uniformity
  • Destination IP Uniformity
  • Source/Destination Ratio
  • Total Flows
  • Unique Source IPs
  • Unique Destination IPs
  • Unique Source Ports
  • Unique Destination Ports

Plot Type:
  • Percentage Difference Bar Charts
  • Shows % change from baseline to polluted state
  • Averaged per epoch group (10 epochs per group)

Color Scheme:
  • Red: Increase from baseline (positive % change)
  • Green: Decrease from baseline (negative % change)
        """
        
        fig.text(0.1, 0.05, summary_text, fontsize=11, 
                verticalalignment='bottom', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✓ PDF generated successfully: {output_pdf}")
    print(f"  Total pages: {len(metrics) + 1}")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generatePlotsEntropy.py <json_file> [output_pdf]")
        print("\nExample:")
        print("  python generatePlotsEntropy.py entropy_epoch_analysis_data_10pct.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        generate_plots(json_path, output_pdf)
    except Exception as e:
        print(f"\nError generating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
