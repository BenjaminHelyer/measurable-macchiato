"""
Visualization of Urysohn's Lemma in 1D (R -> R)

Urysohn's Lemma states that for a normal topological space X and two disjoint
closed sets A and B, there exists a continuous function f: X -> [0,1] such that:
- f(x) = 0 for all x in A
- f(x) = 1 for all x in B

This script visualizes such a function in R with two disjoint closed intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def distance_to_interval(x, interval):
    """Compute the distance from a point x to a closed interval [a, b]."""
    a, b = interval
    if a <= x <= b:
        return 0.0
    elif x < a:
        return a - x
    else:  # x > b
        return x - b

def urysohn_function_1d(x, interval_A, interval_B):
    """
    Construct a Urysohn function in 1D using distance-based approach.
    
    For a point x, the function value is:
    f(x) = d(x, A) / (d(x, A) + d(x, B))
    
    where d(x, A) is the distance from x to interval A.
    This ensures f = 0 on A and f = 1 on B, with smooth transition.
    """
    d_A = distance_to_interval(x, interval_A)
    d_B = distance_to_interval(x, interval_B)
    
    # Handle the case where both distances are zero (shouldn't happen for disjoint sets)
    if d_A == 0:
        return 0.0
    if d_B == 0:
        return 1.0
    
    # Avoid division by zero
    total = d_A + d_B
    if total == 0:
        return 0.5
    
    return d_A / total

def visualize_urysohn_lemma_1d():
    """Create a visualization of Urysohn's Lemma in 1D."""
    # Define two disjoint closed intervals
    # Set A: closed interval [-2, -1]
    interval_A = (-2.0, -1.0)
    
    # Set B: closed interval [1, 2]
    interval_B = (1.0, 2.0)
    
    # Create a fine grid for smooth visualization
    x = np.linspace(-4, 4, 1000)
    
    # Compute the Urysohn function
    print("Computing Urysohn function...")
    f_x = np.array([urysohn_function_1d(xi, interval_A, interval_B) for xi in x])
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot the function
    ax.plot(x, f_x, 'b-', linewidth=2.5, label='Urysohn function f(x)')
    
    # Highlight the sets
    # Set A: where f = 0
    mask_A = (x >= interval_A[0]) & (x <= interval_A[1])
    ax.fill_between(x[mask_A], f_x[mask_A], 0, color='blue', alpha=0.3, 
                    label=f'Set A: [{interval_A[0]}, {interval_A[1]}] (f=0)')
    
    # Set B: where f = 1
    mask_B = (x >= interval_B[0]) & (x <= interval_B[1])
    ax.fill_between(x[mask_B], f_x[mask_B], 1, color='red', alpha=0.3,
                    label=f'Set B: [{interval_B[0]}, {interval_B[1]}] (f=1)')
    
    # Add horizontal lines at y=0 and y=1 for reference
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Mark the intervals on the x-axis
    ax.axvspan(interval_A[0], interval_A[1], alpha=0.1, color='blue')
    ax.axvspan(interval_B[0], interval_B[1], alpha=0.1, color='red')
    
    # Add vertical lines to mark the boundaries
    ax.axvline(x=interval_A[0], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=interval_A[1], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=interval_B[0], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=interval_B[1], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add annotations
    ax.annotate('f(x) = 0', xy=(np.mean(interval_A), 0.05), 
                xytext=(np.mean(interval_A), -0.15), ha='center',
                fontsize=11, fontweight='bold', color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    ax.annotate('f(x) = 1', xy=(np.mean(interval_B), 0.95), 
                xytext=(np.mean(interval_B), 1.15), ha='center',
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('f(x)', fontsize=14)
    ax.set_title('Urysohn\'s Lemma in 1D: f: R → [0,1]', fontsize=16, fontweight='bold')
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate and save the visualization."""
    print("Generating Urysohn's Lemma 1D visualization...")
    fig = visualize_urysohn_lemma_1d()
    
    # Create scratch_results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'scratch_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'urysohn_lemma_1d_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Display the figure
    plt.show()

if __name__ == '__main__':
    main()

