import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data_frame):
    # Plot histograms
    data_frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()
    
    # Plot scatter plot
    sns.scatterplot(
        data=data_frame,
        x="Longitude",
        y="Latitude",
        size="MedHouseVal",
        hue="MedHouseVal",
        palette="viridis",
        alpha=0.5,
    )
    plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    plt.title("Median house value depending of\n their spatial location")
    plt.show()

if __name__ == "__main__":
    from data_loading import load_data
    data = load_data()
    visualize_data(data.frame)