"""Matplotlib styling and a generic publication-quality line plot.

Moved from plottingFunctions.py.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


## Plotting functions

def enable_plt_styling():
    """Enable APS/PRAB-like matplotlib styling for plots.
    """
    # APS / PRAB-like matplotlib configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


## Make a nice plot from arrays

def normalize_arrays(arrays):
    """Normalize input arrays into a list of numpy arrays for plotting utilities."""
    # Case 1: single NumPy array
    if isinstance(arrays, np.ndarray):
        if arrays.ndim == 1:
            return [arrays]          # wrap single dataset
        elif arrays.ndim == 2:
            return list(arrays)      # split into rows
        else:
            raise ValueError("Array must be 1D or 2D")

    # Case 2: iterable of arrays (list/tuple/etc.)
    try:
        return [np.asarray(a) for a in arrays]
    except TypeError:
        raise ValueError("Input must be an array or iterable of arrays")


def make_a_plot(x, y, errs=None, aspect_ratio=0.67, label=r"$\sin(x)$", x_label=r"$x$", y_label=r"$y$", cartesian_axes=[True, True], axes_location=[0, 0], colors=['blue', 'black', 'red']):
    """Create a publication-quality line plot from x/y arrays with optional error bands."""
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.size": 10,
    #     "axes.labelsize": 10,
    #     "axes.titlesize": 10,
    #     "xtick.labelsize": 9,
    #     "ytick.labelsize": 9,
    #     "legend.fontsize": 9,
    #     "axes.linewidth": 0.8,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "xtick.major.size": 4,
    #     "ytick.major.size": 4,
    #     "xtick.minor.size": 2,
    #     "ytick.minor.size": 2,
    #     "xtick.top": True,
    #     "ytick.right": True,
    #     "figure.dpi": 300,
    #     "savefig.dpi": 300,
    # })

    x = normalize_arrays(x)
    y = normalize_arrays(y)
    if errs is not None:
        errs = normalize_arrays(errs)
    nLines = len(x)
    if nLines==1:
        label=[label]
        
    # if len(y_label)==2:
    #     fig, ax1 = plt.subplots(figsize=(3.4, 3.4*aspect_ratio))

    #     ax1.plot(x[0], y[0], lw=1.8, label=label[0], c=colors[0]) if errs is None else ax1.errorbar(x[0], y[0], errs[0], lw=1.8, label=label[0], c=colors[0])
    #     ax1.set_xlabel(x_label)
    #     ax1.set_ylabel(y_label[0])
        
    #     ax2 = ax1.twinx()
    #     ax2.plot(x[1], y[1], lw=1.8, label=label[1], c=colors[1]) if errs is None else ax2.errorbar(x[1], y[1], errs[1], lw=1.8, label=label[1], c=colors[1])
    #     ax2.set_xlabel(x_label)
    #     ax2.set_ylabel(y_label[1])
    
    #     if cartesian_axes[1]:
    #         ax2.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
    #     if cartesian_axes[0]:
    #         ax2.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)

        
    #     # plt.setp(ax1.get_xticklabels(), fontsize=14)
    #     # plt.setp(ax2.get_yticklabels(), fontsize=14)
    #     # plt.setp(ax1.get_yticklabels(), fontsize=14)
        
    #     # Minor ticks
    #     ax2.xaxis.set_minor_locator(AutoMinorLocator())
    #     ax2.yaxis.set_minor_locator(AutoMinorLocator())
    #     # Tick parameters
    #     ax2.tick_params(which="both", width=1)
    #     ax2.tick_params(which="major", length=6)
    #     ax2.tick_params(which="minor", length=3)
        
    #     h1, lab1 = ax1.get_legend_handles_labels()
    #     h2, lab2 = ax2.get_legend_handles_labels()
    #     ax1.legend(h1 + h2, lab1 + lab2, loc='best', frameon=False,handlelength=2.0)

    if not isinstance(y_label, str) and len(y_label) in [2, 3]:  # a list of labels, one per y-axis
    
        fig, ax1 = plt.subplots(
            figsize=(4.5, 4.5 * aspect_ratio)
        )
    
        # -------------------------
        # First Y axis
        # -------------------------
        if errs is None:
            ax1.plot(
                x[0], y[0],
                lw=1.8,
                label=label[0],
                c=colors[0]
            )
        else:
            ax1.errorbar(
                x[0], y[0],
                errs[0],
                lw=1.8,
                label=label[0],
                c=colors[0]
            )
    
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label[0])
    
        # -------------------------
        # Second Y axis
        # -------------------------
        ax2 = ax1.twinx()
    
        if errs is None:
            ax2.plot(
                x[1], y[1],
                lw=1.8,
                label=label[1],
                c=colors[1]
            )
        else:
            ax2.errorbar(
                x[1], y[1],
                errs[1],
                lw=1.8,
                label=label[1],
                c=colors[1]
            )
    
        ax2.set_ylabel(y_label[1])
    
        # -------------------------
        # Third Y axis
        # -------------------------
        if len(y_label) == 3:
    
            ax3 = ax1.twinx()
    
            # Move third axis outward
            ax3.spines["right"].set_position(("outward", 50))
    
            # Make the third spine visible
            ax3.spines["right"].set_visible(True)
    
            if errs is None:
                ax3.plot(
                    x[2], y[2],
                    lw=1.8,
                    label=label[2],
                    c=colors[2]
                )
            else:
                ax3.errorbar(
                    x[2], y[2],
                    errs[2],
                    lw=1.8,
                    label=label[2],
                    c=colors[2]
                )
    
            ax3.set_ylabel(y_label[2])
    
        # -------------------------
        # Cartesian axes
        # -------------------------
        if cartesian_axes[1]:
            ax1.axhline(
                axes_location[1],
                linestyle="--",
                linewidth=1.0,
                color="0.6",
                zorder=0
            )
    
        if cartesian_axes[0]:
            ax1.axvline(
                axes_location[0],
                linestyle="--",
                linewidth=1.0,
                color="0.6",
                zorder=0
            )
    
        # -------------------------
        # Minor ticks
        # -------------------------
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
        if len(y_label) == 3:
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
        # -------------------------
        # Tick parameters
        # -------------------------
        ax1.tick_params(which="both", width=1)
        ax1.tick_params(which="major", length=6)
        ax1.tick_params(which="minor", length=3)
    
        ax2.tick_params(which="both", width=1)
        ax2.tick_params(which="major", length=6)
        ax2.tick_params(which="minor", length=3)
    
        if len(y_label) == 3:
            ax3.tick_params(which="both", width=1)
            ax3.tick_params(which="major", length=6)
            ax3.tick_params(which="minor", length=3)

        # Axis 1
        ax1.spines["left"].set_color(colors[0])
        ax1.tick_params(axis="y", colors=colors[0])
        ax1.yaxis.label.set_color(colors[0])

        # Axis 2
        ax2.spines["right"].set_color(colors[1])
        ax2.tick_params(axis="y", colors=colors[1])
        ax2.yaxis.label.set_color(colors[1])

        # Axis 3
        if len(y_label) == 3:
            ax3.spines["right"].set_color(colors[2])
            ax3.tick_params(axis="y", colors=colors[2])
            ax3.yaxis.label.set_color(colors[2])
    
        # -------------------------
        # Combined legend
        # -------------------------
        h1, lab1 = ax1.get_legend_handles_labels()
        h2, lab2 = ax2.get_legend_handles_labels()
    
        handles = h1 + h2
        labels = lab1 + lab2
    
        if len(y_label) == 3:
            h3, lab3 = ax3.get_legend_handles_labels()
            handles += h3
            labels += lab3
    
        ax1.legend(
            handles,
            labels,
            loc="best",
            frameon=False,
            handlelength=2.0
        )

    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))  # JACoW one-column width
        #fig, ax = plt.subplots(figsize=(6.0, 6.0*aspect_ratio))
        for i, a in enumerate(x):
            ax.plot(1000*x[i], y[i], lw=1.8, label=label[i]) if errs is None else ax.errorbar(x[i], y[i], errs[i], lw=1.8, label=label[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
        if cartesian_axes[1]:
            ax.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        if cartesian_axes[0]:
            ax.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        
        # Minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Tick parameters
        ax.tick_params(which="both", width=1)
        ax.tick_params(which="major", length=6)
        ax.tick_params(which="minor", length=3)
        # Legend
        #loc="upper right",
        ax.legend(loc="best",frameon=False,handlelength=2.0)
        
    # Tight layout for journal export
    fig.tight_layout(pad=0.3)

    # Show explicitly: with interactive mode off (plotMod/print_result turn it off) the inline
    # backend never queues the figure, so it would sit in Gcf until some later plt.show()
    # flushed it into the wrong cell.
    plt.show()

    # Save (recommended formats for journals)
    # fig.savefig("figure.pdf")
    # fig.savefig("figure.eps")
