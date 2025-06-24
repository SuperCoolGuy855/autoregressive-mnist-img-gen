import matplotlib.pyplot as plt
from PIL import Image

# Read images and put into a list
images = []
for i in range(10):
    image = Image.open(f"output_image_{i}.png").convert("L")
    images.append(image)

# Display images in a grid with their number i as title
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, img in zip(axes.flatten(), images):
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # Hide axes
    ax.set_title(f"{images.index(img)}")
plt.tight_layout()
plt.savefig("output_images_grid.png")