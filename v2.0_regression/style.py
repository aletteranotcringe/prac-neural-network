import matplotlib.pyplot as plt

def apply_dark_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        'axes.facecolor': '#3a414d',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'figure.facecolor': '#3a414d',
        'figure.edgecolor': 'black',
        'grid.color': '#555555'
    })
