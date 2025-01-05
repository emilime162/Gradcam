
from __future__ import print_function

import os.path as osp
import os
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

from grad_cam import (

    GradCAM,
  
)


def create_grid(raw_image, gradcam_results, layer_names, output_path):
    """
    Create a 2x4 grid with the input image and Grad-CAM visualizations.

    Args:
        raw_image (PIL.Image): Original input image.
        gradcam_results (list of numpy arrays): Grad-CAM heatmaps for each layer.
        layer_names (list of str): Names of the layers visualized.
        output_path (str): Path to save the generated grid.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    

    # Original image in the top-left
    # Remove the batch dimension


    axes[0, 0].imshow(raw_image[0])
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original Picture")
    
    # Grad-CAM results in the remaining cells
    for i, (heatmap, layer_name) in enumerate(zip(gradcam_results, layer_names)):
        row, col = divmod(i + 1, 4)
        axes[row, col].imshow(heatmap, cmap="jet", alpha=0.7)  # Heatmap
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Layer {layer_name}")
    
    # Save and show the grid
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(float) + raw_image.astype(float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)



@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM for VGG16 with dynamically predicted classes
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.vgg16(pretrained=True)
    model.to(device)
    model.eval()

    # Target layers for Grad-CAM
    layers_to_visualize = [14, 17, 19, 21, 24, 26, 28]
    target_layers = [f"features.{i}" for i in layers_to_visualize]

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    predicted_class = ids[:, 0]  # Get the class with the highest probability
    print(f"Predicted class: {predicted_class.item()} ({classes[predicted_class.item()]})")
    predicted_class = predicted_class.view(-1, 1)  # Shape: [B, 1]

    gcam.backward(ids=predicted_class)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    gradcam_results = []

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)
        gradcam_results.append(regions[0][0])  # Heatmap for the first image


        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[predicted_class.item()], float(probs[j, predicted_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "vgg16-predicted", target_layer, classes[predicted_class.item()]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

    # Create and save the grid
    output_path = f"{output_dir}/gradcam_grid.png"
    create_grid(raw_images, gradcam_results, layers_to_visualize, output_path)
    # Remove the batch dimension

    print(f"Grid saved at {output_path}")




if __name__ == "__main__":
    main()
