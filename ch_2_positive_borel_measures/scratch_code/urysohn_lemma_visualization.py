"""
Visualization of Urysohn's Lemma

Urysohn's Lemma states that for a normal topological space X and two disjoint
closed sets A and B, there exists a continuous function f: X -> [0,1] such that:
- f(x) = 0 for all x in A
- f(x) = 1 for all x in B

This script visualizes such a function in R^2 with two disjoint closed sets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
import os


def urysohn_function_vectorized(X, Y, set_A, set_B):
    """
    Construct a Urysohn function using distance-based approach (vectorized).
    
    For a point (x, y), the function value is:
    f(x, y) = d(x, A) / (d(x, A) + d(x, B))
    
    where d(x, A) is the distance from x to set A.
    This ensures f = 0 on A and f = 1 on B, with smooth transition.
    """
    # Reshape grids to vectors
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute distances to both sets (vectorized)
    if len(set_A) > 0:
        dists_A = cdist(points, set_A)
        d_A = np.min(dists_A, axis=1)
    else:
        d_A = np.full(points.shape[0], np.inf)
    
    if len(set_B) > 0:
        dists_B = cdist(points, set_B)
        d_B = np.min(dists_B, axis=1)
    else:
        d_B = np.full(points.shape[0], np.inf)
    
    # Compute the Urysohn function
    # Handle edge cases
    result = np.zeros_like(d_A)
    
    # Points in set A (distance = 0)
    mask_A = d_A == 0
    result[mask_A] = 0.0
    
    # Points in set B (distance = 0)
    mask_B = d_B == 0
    result[mask_B] = 1.0
    
    # Points not in either set
    mask_other = ~(mask_A | mask_B)
    total = d_A[mask_other] + d_B[mask_other]
    result[mask_other] = d_A[mask_other] / total
    
    # Reshape back to grid
    return result.reshape(X.shape)

def create_closed_sets():
    """
    Create two disjoint closed sets in R^2.
    We'll use two circles as examples.
    """
    # Set A: circle centered at (-1, 0) with radius 0.5
    center_A = np.array([-1.0, 0.0])
    radius_A = 0.5
    
    # Set B: circle centered at (1, 0) with radius 0.5
    center_B = np.array([1.0, 0.0])
    radius_B = 0.5
    
    # Sample points on the boundaries and interiors of the circles
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Points on and inside circle A
    set_A = []
    for r in np.linspace(0, radius_A, 20):
        for t in theta[::5]:  # Sample every 5th angle
            point = center_A + r * np.array([np.cos(t), np.sin(t)])
            set_A.append(point)
    set_A = np.array(set_A)
    
    # Points on and inside circle B
    set_B = []
    for r in np.linspace(0, radius_B, 20):
        for t in theta[::5]:
            point = center_B + r * np.array([np.cos(t), np.sin(t)])
            set_B.append(point)
    set_B = np.array(set_B)
    
    return set_A, set_B, center_A, radius_A, center_B, radius_B

def visualize_urysohn_lemma():
    """Create a comprehensive visualization of Urysohn's Lemma."""
    # Create the disjoint closed sets
    set_A, set_B, center_A, radius_A, center_B, radius_B = create_closed_sets()
    
    # Create a grid for visualization
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Compute the Urysohn function on the grid (vectorized for efficiency)
    print("Computing Urysohn function on grid...")
    Z = urysohn_function_vectorized(X, Y, set_A, set_B)
    
    # Create the visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Heatmap with contour lines
    ax1 = axes[0]
    im = ax1.contourf(X, Y, Z, levels=50, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax1.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # Draw the sets
    circle_A = Circle(center_A, radius_A, color='blue', fill=True, alpha=0.7, label='Set A (f=0)')
    circle_B = Circle(center_B, radius_B, color='red', fill=True, alpha=0.7, label='Set B (f=1)')
    ax1.add_patch(circle_A)
    ax1.add_patch(circle_B)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Urysohn Function: Heatmap View', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='Function Value f(x,y)')
    
    # Right plot: 3D surface
    ax2 = axes[1]
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.9, 
                           linewidth=0, antialiased=True, vmin=0, vmax=1)
    
    # Draw the sets on the 3D plot
    theta_3d = np.linspace(0, 2 * np.pi, 50)
    for r in [radius_A, radius_B]:
        center = center_A if r == radius_A else center_B
        x_circle = center[0] + r * np.cos(theta_3d)
        y_circle = center[1] + r * np.sin(theta_3d)
        z_circle = np.zeros_like(x_circle) if r == radius_A else np.ones_like(x_circle)
        ax2.plot(x_circle, y_circle, z_circle, 
                color='black' if r == radius_A else 'black', linewidth=2)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('f(x,y)', fontsize=12)
    ax2.set_title('Urysohn Function: 3D Surface View', fontsize=14, fontweight='bold')
    ax2.set_zlim(0, 1)
    fig.colorbar(surf, ax=ax2, shrink=0.5, label='Function Value')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate and save the visualization."""
    print("Generating Urysohn's Lemma visualization...")
    fig = visualize_urysohn_lemma()
    
    # Create scratch_results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'scratch_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'urysohn_lemma_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Display the figure
    plt.show()

if __name__ == '__main__':
    main()

