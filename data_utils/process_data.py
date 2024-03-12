import matplotlib.pyplot as plt
from torchvision import transforms
import os


def preprocess(img):
    pass


def postprocess(prediction, shape):
    pass


def plot_images_with_masks(dataset, indices, num_images_per_row=2, 
                           save=False, save_path="./results/plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(10, 5*num_rows))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        ax = axes[i // num_images_per_row, i % num_images_per_row] if num_rows > 1 else axes[i % num_images_per_row]
        ax.imshow(transforms.ToPILImage()(img))
        ax.imshow(mask, alpha=0.5, cmap='jet')
        ax.set_title(f'Image {idx}')
        ax.axis('off')

    plt.tight_layout()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'image_masks_plot.png'))
    
    plt.show()