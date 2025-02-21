import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from YOLOX.yolox.data import TrainTransformV3

# Assuming TrainTransformV2 and other necessary imports are already included here


def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for display purposes
    return image


def visualize_image_with_boxes(ax, image, boxes, title="Image"):
    """Visualize an image with bounding boxes on a matplotlib axis."""
    # Check if the image is in channel-first format and adjust if necessary
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)  # CHW to HWC
    ax.imshow(image.astype(np.uint8))
    ax.set_title(title)


def main(image_path):
    image = load_image(image_path)
    # Example bounding boxes [x1, y1, x2, y2]
    boxes = np.array([[50, 50, 200, 200], [150, 150, 300, 300]])
    labels = np.array([1, 2])  # Dummy labels

    transform = TrainTransformV3(enable_weather_transform=True, max_labels=50)

    fig, axes = plt.subplots(
        2, 4, figsize=(15, 10)
    )  # Adjust subplot grid for 10 images
    axes = axes.ravel()

    for i in tqdm.tqdm(range(8)):
        # Repeatedly apply transformation to visualize different results
        transformed_image, transformed_data = transform(
            image.copy(), np.hstack((boxes, labels[:, None])), (640, 640)
        )
        transformed_boxes = transformed_data[:, 1:5]  # Extract transformed boxes

        # Visualize each transformed image and its boxes
        visualize_image_with_boxes(
            axes[i],
            transformed_image,
            transformed_boxes,
            title=f"Transformed Image {i + 1}",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "../../test-yoloxv2_validation-step/images/train2017/images/018fe72d-a345-7fae-99ce-fb8cc94bd5bb.JPG"
    main(image_path)
