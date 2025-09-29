"""
PHQ-8 Binary Classification Grid Visualization Creator (§5.2 & §6)

Purpose: Creates 2x2 grid visualizations combining PHQ-8 binary classification plots
         by gender and depression status for comprehensive demographic analysis.
         Supports both XY scatter plots and step-wise trajectory visualizations.

Paper Section: §5.2 PHQ-8 Results & §6 Discussion (demographic analysis)
Application: Visual analysis of depression prediction patterns across gender groups

This script generates composite visualizations for PHQ-8 binary classification analysis:

1. Grid Layout Organization:
   - Top Left: Female, PHQ Binary = 0 (Non-depressed)
   - Top Right: Male, PHQ Binary = 0 (Non-depressed)
   - Bottom Left: Female, PHQ Binary = 1 (Depressed)
   - Bottom Right: Male, PHQ Binary = 1 (Depressed)

2. Visualization Types Supported:
   - XY Plots: Scatter plot visualizations of VA coordinates
   - Step Plots: Temporal trajectory analysis showing VA progression
   - Configurable plot type selection via plot_type variable

3. Demographic Analysis Features:
   - Gender-stratified visualization (Male vs Female)
   - Depression status comparison (Binary 0 vs 1)
   - Side-by-side layout for direct visual comparison
   - Standardized image dimensions for consistency

4. Technical Implementation:
   - PIL-based image composition and grid creation
   - Automatic image sizing and positioning
   - Configurable base path for input image location
   - Error handling for invalid plot type selection

Research Applications:
- Visual analysis of gender differences in depression-related VA patterns
- Comparison of emotional trajectories between depressed and non-depressed groups
- Demographic stratification for clinical interpretation
- Publication-ready composite figures for paper illustrations

Input Images Expected:
- Female_PHQ_Binary_0.png / Female_PHQ_Binary_0_steps.png
- Male_PHQ_Binary_0.png / Male_PHQ_Binary_0_steps.png  
- Female_PHQ_Binary_1.png / Female_PHQ_Binary_1_steps.png
- Male_PHQ_Binary_1.png / Male_PHQ_Binary_1_steps.png

Output:
- combined_PHQ_binaries_grid_xy.png (XY scatter grid)
- combined_PHQ_binaries_grid_step.png (temporal trajectory grid)

Clinical Insights Supported:
- Gender-specific emotional expression patterns in depression
- VA trajectory differences between clinical and non-clinical populations
- Temporal dynamics of emotional states in depressed vs healthy individuals
- Visual validation of VA-based depression classification effectiveness

Usage:
    # Configure plot type and run
    plot_type = "xy"      # or "step" for temporal analysis
    base_path = "/path/to/images/"
    python create_grid.py
    
    # Generates: combined_PHQ_binaries_grid_xy.png

Related Files:
- va_classifier.data.visualize_xy_va — Individual VA scatter plot generation
- va_classifier.data.visualize_time_step — Temporal trajectory visualization
- va_classifier.evaluation — Performance metrics and analysis tools

Note: This visualization tool supports the paper's demographic analysis by enabling
      direct visual comparison of VA patterns across gender and depression status.
"""

from PIL import Image

base_path = "PATH"
plot_type = "step"

plot_images = {
    "xy": [
        "Female_PHQ_Binary_0.png",
        "Male_PHQ_Binary_0.png",
        "Female_PHQ_Binary_1.png",
        "Male_PHQ_Binary_1.png"
    ],
    "step": [
        "Female_PHQ_Binary_0_steps.png",
        "Male_PHQ_Binary_0_steps.png",
        "Female_PHQ_Binary_1_steps.png",
        "Male_PHQ_Binary_1_steps.png"
    ]
}

image_files = plot_images.get(plot_type)
if image_files is None:
    raise ValueError(f"Unknown plot_type: {plot_type}. Choose 'xy' or 'step'.")

images = [Image.open(base_path + fname) for fname in image_files]
img_width, img_height = images[0].size

grid_img = Image.new('RGB', (2 * img_width, 2 * img_height))

positions = [
    (0, 0), (img_width, 0),
    (0, img_height), (img_width, img_height)
]
for img, pos in zip(images, positions):
    grid_img.paste(img, pos)

output_path = base_path + f"combined_PHQ_binaries_grid_{plot_type}.png"
grid_img.save(output_path)
print(f"Saved 2x2 {plot_type} grid image to {output_path}")
