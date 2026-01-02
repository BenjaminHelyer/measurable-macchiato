"""
Visualization of Urysohn's Lemma in 1D (R -> R) with a Weird Metric

Urysohn's Lemma states that for a normal topological space X and two disjoint
closed sets A and B, there exists a continuous function f: X -> [0,1] such that:
- f(x) = 0 for all x in A
- f(x) = 1 for all x in B

This script visualizes such a function in R with two disjoint closed intervals,
but using a non-standard metric: d(x,y) = |arctan(x) - arctan(y)|

This metric compresses the real line non-linearly, making large values appear
closer together than they would under the Euclidean metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def weird_metric(x, y):
    """
    A weird metric on R: d(x,y) = |arctan(x) - arctan(y)|
    
    This metric:
    - Compresses large values (as |x| -> inf, arctan(x) -> ±π/2)
    - Is translation-invariant in the arctan space
    - Makes the real line behave as if it's bounded
    """
    return np.abs(np.arctan(x) - np.arctan(y))

def distance_to_interval_weird_metric(x, interval, metric_func):
    """
    Compute the distance from a point x to a closed interval [a, b]
    using the weird metric.
    """
    a, b = interval
    if a <= x <= b:
        return 0.0
    elif x < a:
        return metric_func(x, a)
    else:  # x > b
        return metric_func(x, b)

def urysohn_function_weird_metric(x, interval_A, interval_B, metric_func):
    """
    Construct a Urysohn function in 1D using the weird metric.
    
    For a point x, the function value is:
    f(x) = d(x, A) / (d(x, A) + d(x, B))
    
    where d(x, A) is the distance from x to interval A using the weird metric.
    This ensures f = 0 on A and f = 1 on B, with smooth transition.
    """
    d_A = distance_to_interval_weird_metric(x, interval_A, metric_func)
    d_B = distance_to_interval_weird_metric(x, interval_B, metric_func)
    
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

def visualize_urysohn_lemma_weird_metric():
    """Create a visualization of Urysohn's Lemma in 1D with weird metric."""
    # Define two disjoint closed intervals
    # Set A: closed interval [-2, -1]
    interval_A = (-2.0, -1.0)
    
    # Set B: closed interval [1, 2]
    interval_B = (1.0, 2.0)
    
    # Create a fine grid for smooth visualization (extend further to show weird behavior)
    x = np.linspace(-10, 10, 2000)
    
    # Compute the Urysohn function with weird metric
    print("Computing Urysohn function with weird metric...")
    f_x_weird = np.array([urysohn_function_weird_metric(xi, interval_A, interval_B, weird_metric) 
                          for xi in x])
    
    # Also compute with standard metric for comparison
    def euclidean_metric(x, y):
        return abs(x - y)
    
    def distance_to_interval_euclidean(x, interval):
        a, b = interval
        if a <= x <= b:
            return 0.0
        elif x < a:
            return a - x
        else:
            return x - b
    
    f_x_euclidean = np.array([
        urysohn_function_weird_metric(xi, interval_A, interval_B, euclidean_metric)
        for xi in x
    ])
    
    # Create the visualization with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Weird metric
    ax1 = axes[0]
    ax1.plot(x, f_x_weird, 'purple', linewidth=2.5, label='Urysohn function with weird metric')
    
    # Highlight the sets
    mask_A = (x >= interval_A[0]) & (x <= interval_A[1])
    ax1.fill_between(x[mask_A], f_x_weird[mask_A], 0, color='blue', alpha=0.3, 
                    label=f'Set A: [{interval_A[0]}, {interval_A[1]}] (f=0)')
    
    mask_B = (x >= interval_B[0]) & (x <= interval_B[1])
    ax1.fill_between(x[mask_B], f_x_weird[mask_B], 1, color='red', alpha=0.3,
                    label=f'Set B: [{interval_B[0]}, {interval_B[1]}] (f=1)')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvspan(interval_A[0], interval_A[1], alpha=0.1, color='blue')
    ax1.axvspan(interval_B[0], interval_B[1], alpha=0.1, color='red')
    
    ax1.axvline(x=interval_A[0], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=interval_A[1], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=interval_B[0], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=interval_B[1], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('f(x)', fontsize=14)
    ax1.set_title('Urysohn\'s Lemma with Weird Metric: d(x,y) = |arctan(x) - arctan(y)|', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Bottom plot: Standard Euclidean metric for comparison
    ax2 = axes[1]
    ax2.plot(x, f_x_euclidean, 'b-', linewidth=2.5, label='Urysohn function with Euclidean metric')
    
    mask_A = (x >= interval_A[0]) & (x <= interval_A[1])
    ax2.fill_between(x[mask_A], f_x_euclidean[mask_A], 0, color='blue', alpha=0.3, 
                    label=f'Set A: [{interval_A[0]}, {interval_A[1]}] (f=0)')
    
    mask_B = (x >= interval_B[0]) & (x <= interval_B[1])
    ax2.fill_between(x[mask_B], f_x_euclidean[mask_B], 1, color='red', alpha=0.3,
                    label=f'Set B: [{interval_B[0]}, {interval_B[1]}] (f=1)')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvspan(interval_A[0], interval_A[1], alpha=0.1, color='blue')
    ax2.axvspan(interval_B[0], interval_B[1], alpha=0.1, color='red')
    
    ax2.axvline(x=interval_A[0], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=interval_A[1], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=interval_B[0], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=interval_B[1], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('f(x)', fontsize=14)
    ax2.set_title('Comparison: Standard Euclidean Metric d(x,y) = |x - y|', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate and save the visualization."""
    print("Generating Urysohn's Lemma visualization with weird metric...")
    fig = visualize_urysohn_lemma_weird_metric()
    
    # Create scratch_results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'scratch_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'urysohn_lemma_weird_metric_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Display the figure
    plt.show()

if __name__ == '__main__':
    main()

