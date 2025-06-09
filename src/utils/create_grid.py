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
