# Plotter for Checking Absolute Gradient of Each Layer
Visualizing gradient flow of each layer to check whether a network is really learning. A very small |gradient| indicates potentially poor learning.
Further information can be found at [this pytorch discussion](https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063)

Features of this repository:
   * A class to allow simple call and record of gradient history (if one needs post-check).
   * Supports figure updating for real-time monitoring in Jupyer notebook or VS code interactive window.
   * Latter plotted data is jittered to the right and is less transparent to reserve time info.
   * Tracks mean, standard deviation, min, median, and max values on each layer
   * Supports non-default percentile to track
   
This repo is revised from https://github.com/alwynmathew/gradflow-check for further functionality

## Usage
```
...
gfp = GradFlowPlotter(model, total_epoch=total_epoch) # total_epoch needs to be >= # of time calling update_grad (it needs not to be exactly #epoch)
for ep in range(total_epoch):
      ...
      loss.backward()
      gfp.update_grad(model, ep) # gather new gradient values
      gfp.update_plot(ep) # plot for every update in Jupyter
      ...
gfp.plot_all() # use this function rather than gfp.update_plot(ep) to save time if only a summary plot is needed
```
Below is what one would see if using gfp.update_plot() in Jupyter notebook or VS code interactive window. Using gfp.plot_all() will only generate a static figure with all the avaialble gradient history.
![demo](https://github.com/nafraw/gradflow-plotter/assets/10406101/dd8adca9-c98f-489b-b435-c611f0e72850)
