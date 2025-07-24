import matplotlib.pyplot as plt

def show_sample_image(image, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["accuracy"], label="train accuracy")
    ax.plot(history.history["val_loss"], label="validation loss")
    ax.plot(history.history["val_accuracy"], label="validation accuracy")
    ax.legend()
    plt.show()