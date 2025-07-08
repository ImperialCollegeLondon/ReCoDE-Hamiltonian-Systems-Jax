# Using Matplotlib

This exemplar uses Matplotlib to produce outputs in the `plot_trajectories` methods of the `HarmonicOscillator` and `NBody` classes. Most of this is fairly standard, but there are a few things to note.

## Passing an Axes Object as an Argument

Both `plot_trajectory` method taken an `Axes` object as an argument. This is a common pattern in Matplotlib, where you create a figure and axes object, and then pass the axes object to the plotting function. This allows the code which calls the functions to control the figure the axes appear in, and other context in which the axes are placed. In the notebooks showcasing the simple harmonic oscillator and N-body systems, we create a figure and axes object using `plt.subplots()` and then pass the axes object to the `plot_trajectory` method.

## Synchronising Colours

In the `plot_trajectory` methods in the `HarmonicOscillator` and `NBody` classes, there are lines that show the trajectory over time, and a point at the end of the trajectory. In order to ensure that the point at the end of the trajectory has the same colour as the line, we save a reference to the value returned from the call to `ax.plot()` in a variable called `line`. This is a list of line objects, and we can access the first element of this list to get the line object itself. We then use the `get_color()` method of the line object to get the colour of the line, which we then use to set the colour of the point at the end of the trajectory. This ensures that the point at the end of the trajectory has the same colour as the line.