import numpy as np
import matplotlib.pyplot as plt

def visualise_dataset(data_x, data_y, samples_per_class: int = 3):
    plot_index = 0
    number_of_classes = len(np.unique(data_y))
    fig, axes = plt.subplots(samples_per_class, number_of_classes)
    flaten_axes = axes.flatten()
    for example_index in range(samples_per_class):
        for class_index in range(number_of_classes):
            image = data_x[data_y == class_index][example_index]
            flaten_axes[plot_index].imshow(image.astype(np.uint8))
            flaten_axes[plot_index].axis('off')
            flaten_axes[plot_index].set_title(f"{class_index}")
            plot_index += 1
    plt.show()