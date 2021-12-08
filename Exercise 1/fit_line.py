import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


fig, ax1 = plt.subplots()
coords = []


def my_linfit(x, y):
    """
    Linear equation solver
    :param: x 
    :param: y 
    :return: a (slope) and b (intercept) of line
    """
    ySummation = sum(y)/len(y)
    xSummation = sum(x)/len(x)
    xySummation = sum(x*y)/len(x)
    xSqareSummation = sum(x*x)/len(x)

    b=  ((ySummation*xSqareSummation) - (xySummation*xSummation)) / (xSqareSummation - (xSummation*xSummation))
    a = (xySummation - (b * xSummation))/xSqareSummation     
    return a,b


def onclick(event):
    """
    mouse click function to store coordinates and plot markers
    left click to add points
    right click to stop collecting points and display linear regression line
    """
    if not event.inaxes == ax1:
        return
    if event.button == 1:
        plot_points(event)
    elif event.button == 3:
        plt.close()


def plot_points(event):
    """
    plotting markers where clicks occur
    """
    ix, iy = event.xdata, event.ydata
    ax1.plot(ix, iy, linestyle='--', marker='x', color='r')
    fig.canvas.draw_idle()
    coords.append((ix, iy))


def main():
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.axis([-2, 5, 0, 3])
    plt.show()
    x = np.array([])
    y = np.array([])
    for ix, iy in coords:
        x = np.append(x, ix)
        y = np.append(y, iy)

    a, b = my_linfit(x, y)
    a1,b1 = np.polyfit(x,y,1)
    xp = np.arange(-2, 5, 0.1)
    title = f'Fitted Line Parameters: a={round(a,4)} and b={round(b,4)} versus a1={round(a1,4)} and b1={round(b1,4)}'
    plt.title(title)
    plt.plot(xp, (a*xp)+b, 'r-')
    plt.plot(xp, (a1*xp)+b1, 'b--')
    plt.plot(x, y, 'rx')
    plt.axis([-2, 5, 0, 3])
    plt.show()

if __name__ == "__main__":
    main()