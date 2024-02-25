import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy.stats as stats # if one wants to add other different statistics

# is_notebook() is used to detect whether it makes sense to "update plot" on-the-fly
# ref: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

class GradFlowPlotter:
    ## Gradient flow plotter revised from:
    # https://github.com/alwynmathew/gradflow-check
    # This class is used to plot statistics of |gradient| in each layer to check whether a network is really learning
    # Features of this class:
    #   A class to allow simple call and handle of gradient history.
    #   Supports figure updating for real-time monitoring in Jupyer notebook or VS code interactive window.
    #   Latter plotted data is jittered to the right and is less transparent to reserve time info.
    #   Tracks mean, standard deviation, min, median, and max values on each layer
    #   Supports non-default percentile to track
    def __init__(self, model, total_epoch, fignum_stat=None, fignum_prc=None,
                 figsize_stat = None, figsize_prc = None,
                 color=['g', 'b', 'y', 'k'], target_prctile = [0, 50, 100]) -> None:
        # model: deep learning model
        # total_epoch: how many epoch in optimization OR how many time one would like to check gradient.
        #              This value can be larger than or equal to, but NOT smaller than, the actual needed #.
        # color: cyclic color to be used for each plotting
        # target_prctile: which percentile to trace
        # fignum_stat, fignum_prc: which fignum to be used when calling plt.figure
        # figsize_stat, figsize_prc: decide figure sizes in case the auto-generated ones do not work well
        self.target_prctile = target_prctile # suggest to have 0 and 100 for the min and max values
        self.layers_name = []
        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                self.layers_name.append(n)
        self.nLayer = len(self.layers_name)
        self.prctile = np.full((self.nLayer, len(target_prctile), total_epoch), np.nan)
        self.stat = np.full((self.nLayer, 2, total_epoch), np.nan) # mean and std
        self.stat_name = ['mean', 'std']
        self.nEpoch = total_epoch
        self.color = color # color to cyclic with
        scale = 0.5
        subx = (np.array(range(total_epoch))-total_epoch//2)/total_epoch*scale # to adjust x based on epoch
        self.x = np.tile(range(self.nLayer), [total_epoch, 1]).T
        self.plot_x = self.x + subx[np.newaxis, :] # later epoch will be jittered to the right side
        ## figure related initialization
        self.fig_prc = plt.figure(fignum_prc)
        self.fig_stat = plt.figure(fignum_stat)
        if is_notebook():
            self.hfig_prc = display(self.fig_prc, display_id=True)
            self.hfig_stat = display(self.fig_stat, display_id=True)
        self.set_figsize(figsize_stat=figsize_stat, figsize_prc=figsize_prc)
        plt.figure(figsize=self.figsize_stat)
        plt.figure(figsize=self.figsize_prc)

    def update_grad(self, model, epoch):
        lid = 0
        for l, (n, p) in enumerate(model.named_parameters()):
            if(p.requires_grad) and ("bias" not in n):
                abs_grad = p.grad.abs().cpu().numpy()
                self.prctile[lid, :, epoch] = np.percentile(abs_grad, self.target_prctile)
                self.stat[lid, :, epoch] = [np.mean(abs_grad), np.std(abs_grad)]
                lid+=1

    def update_plot(self, epoch):
        # Jupyter notebook or VS code interactive window does not always update plot on-the-fly
        # This function is designed to tackle the issue.
        assert is_notebook(), "This function is supposed to work only in a Jupyter notebook or interactive window, not terminal"
        self.plot_all(target_epoch=epoch)
        self.fig_prc.canvas.draw()
        self.fig_stat.canvas.draw()
        self.hfig_prc.update(self.fig_prc)
        self.hfig_stat.update(self.fig_stat)

    def get_suplot_size(self, nPlot):
        # make subplot layout automatic and to be square-like
        nRow = int(np.floor(np.sqrt(nPlot)))
        nCol = nRow
        if nRow * nCol < nPlot:
            nCol += 1
        if nRow * nCol < nPlot:
            nRow += 1
        return nRow, nCol

    def _set_figsize(self, nSubplot, figsize):
        nRow, nCol = self.get_suplot_size(nSubplot)
        if figsize is None:
            base_figsize = (0.2*self.nLayer, self.nLayer*0.15)
            figsize = (base_figsize[0]*nCol, base_figsize[1]*nRow)
        return figsize, nRow, nCol

    def set_figsize(self, figsize_stat=None, figsize_prc=None):
        # figsize is suggested to be bigger in y direction because vertical names (xtick labels) are long
        self.figsize_stat, nR, nC = self._set_figsize(nSubplot=self.stat.shape[1], figsize=figsize_stat)
        self.stat_nR_nC = [nR, nC]
        self.figsize_prc, nR, nC = self._set_figsize(nSubplot=len(self.target_prctile), figsize=figsize_prc)
        self.prc_nR_nC = [nR, nC]

    def plot_all(self, target_epoch=None):
        plt.figure(self.fig_prc)
        if target_epoch is None:
            target_epoch = np.array(range(self.nEpoch))
        x = self.plot_x[:, target_epoch]
        for pi in range(len(self.target_prctile)):
            plt.subplot(self.prc_nR_nC[0], self.prc_nR_nC[1], pi+1)
            clr = self.color[pi%(len(self.color))] # cyclic around available colors
            data = self.prctile[:, pi, target_epoch]
            self.plot_one_stat(x, data, color=clr, epoch=target_epoch)
            self.plot_annotation(ylim_top=np.nanmax(self.prctile))
            plt.title(f'{self.target_prctile[pi]}-th perecentile')
        plt.tight_layout()

        plt.figure(self.fig_stat)
        for si in range(self.stat.shape[1]):
            plt.subplot(self.stat_nR_nC[0], self.stat_nR_nC[1], si+1)
            clr = self.color[si%(len(self.color))] # cyclic around available colors
            data = self.stat[:, si, target_epoch]
            self.plot_one_stat(x, data, color=clr, epoch=target_epoch)
            self.plot_annotation(ylim_top=np.nanmax(self.stat))
            plt.title(self.stat_name[si])
        plt.tight_layout()

    def plot_one_stat(self, x, data, color, epoch): # plot for one stat, either percentile or mean/std
        # later epoch has a darker color (smaller alpha = transparent)
        alpha = (epoch/self.nEpoch + 0.1)/1.1 # make sure the lowest is 0.1
        if x.ndim == 2:        
            for i in range(x.shape[1]): # because a separate alpha is not supported for each data
                sns.scatterplot(x=x[:,i], y=data[:, i], color=color, alpha=alpha[i])
        else:
            sns.scatterplot(x=x, y=data, color=color, alpha=alpha)
        plt.hlines(0, -1, self.nLayer+1, lw=2, color="k")

        # NOTE:
            # The lines below are ideally what should happen without the for-loop,
            # but the alpha will not be properly used when plotting multiple epoch.
            # Plus, a warning,
            # kws["alpha"] = 1 if self.alpha == "auto" else self.alpha, is thrown
            # when calling this function with multiple epoch because seaborn seems not handling
            # multi-value alpha assigned manually. Although it still plots, but not doing what
            # was expected.
            # Another maybe related issue below, but not relevant here because removing color
            # does not change anything.
            # ref: https://github.com/mwaskom/seaborn/issues/1966
        # if x.ndim > 1:
        #     alpha = np.tile(alpha, [x.shape[0], 1]).flatten()
        # sns.scatterplot(x=x.flatten(), y=data.flatten(), color=color, alpha=alpha)

    def plot_annotation(self, ylim_top=0.002):
        plt.xticks(range(0, self.nLayer, 1), self.layers_name, rotation="vertical")
        plt.xlim(left=-0.5, right=self.nLayer-0.5)
        plt.ylim(bottom = -0.001, top=ylim_top) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("gradient stats")
        plt.title("Gradient flow")
        plt.grid(True)


## A demo of how to use
if __name__ == '__main__':
    ## import necessary libraries for demo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    ## generate a random architecture and data
    nClass, nBatch, nFeature = 2, 10, 1000
    model = nn.Sequential(
        nn.Linear(nFeature, 100),
        nn.ELU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, nClass),
        nn.Softmax(dim=0)
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    ## initialize GradFlowPlotter,
    # Although it is called total_epoch, it is actually total # of calling update_grad.
    # So, it can be total_epoch*total_batch if one wants to record everything. If one has
    # no idea how total_epoch*total_batch should be, make a number that is bigger than that
    total_epoch = 10
    gfp = GradFlowPlotter(model, total_epoch=total_epoch*2) # *2 on purpose, to demo that it still works

    ## train on random data
    model.train()
    i = 0
    for ep in range(total_epoch):
        X = torch.rand((nBatch, nFeature)).to(device)
        y = torch.rand((nBatch, nClass)) > 0.5
        y = y.to(device)
        nBatch = X.shape[0]
        y_pred = model(X)
        loss = loss_func(y_pred, y.float())
        i += nBatch
        optim.zero_grad()
        loss.backward()
        # after calling backward
        gfp.update_grad(model, ep) # gather new gradient values
        gfp.update_plot(ep) # plot for every update, comment it out if not desired
        optim.step()
    # NOTE: it is for demo purpose that one calls update_plot in every loop and plot_all after looping is finished
        # One only needs to choose one in general. Also, one might notice that the figures from both calls might have
        # different alpha values. Specifically, plot_all() looks more opaque. This is because both functions plots on
        # the same figures and causing the latter plotted one looks darker due to overlapping of the same scatterplot.
    gfp.plot_all() # plot all at once