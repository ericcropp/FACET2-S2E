"""Phase-space and slice plots of a ParticleGroup.

Moved from UTILITY_plotMod.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pmd_beamphysics
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from copy import copy


def plotMod(particle_group, key1='t', key2='p', 
                  bins=None,
                  *,
                  xlim=None,
                  ylim=None,
                  tex=True,
                  nice=True,
                  fig=None,
                  outer=None,
                  i=None,
                  z_from_t=False,
                  background=None,
                  **kwargs):
    """
    Create a 2D hexbin plot with marginal histograms for a ParticleGroup, derived from openPMD-beamphysics marginal_plot().

    Parameters
    ----------
    particle_group : ParticleGroup
        The particle group to plot.
    key1 : str, default 't'
        The key for the x-axis variable.
    key2 : str, default 'p'
        The key for the y-axis variable.
    bins : int, optional
        Number of bins for the histograms and hexbin. If None, determined automatically from particle count.
    xlim : tuple, optional
        Limits for the x-axis.
    ylim : tuple, optional
        Limits for the y-axis.
    tex : bool, default True
        Whether to use TeX-style labels.
    nice : bool, default True
        Whether to use nice units and scaling for axes.
    fig, outer, i : Figure, GridSpec, int, optional
        If all three are given, the plot is drawn into cell ``outer[i]`` of an existing figure
        (used by print_result to put several plots side by side). Otherwise a new figure is created.
    z_from_t : bool, default False
        If True, a key 'z' is plotted as z = -c * delta_t. Useful for beams recorded at a fixed s
        (e.g. Bmad output), where all particles share the same z and differ in t.
    background : color, optional
        Background color of the joint plot. Default: the lowest colormap color.
    **kwargs : dict
        Additional keyword arguments passed to plt.figure() when a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.

    Notes
    -----
    - When a new figure is created, all previous figures are closed and interactive plotting is
      left disabled, so that display(plotMod(...)) shows the figure exactly once.
    - When drawing into an existing figure, nothing is closed and the interactive state is restored.
    - The joint plot shows a weighted hexbin of the selected variables, with marginal histograms above and to the right.
    - Axes are labeled with units and optionally TeX formatting.
    - Intended for quick visualization of beam phase space projections.
    """    

    embedded = fig is not None and outer is not None and i is not None

    if embedded:
        was_interactive = plt.isinteractive()
    else:
        plt.close('all')
    
    CMAP0 = copy(plt.get_cmap('viridis'))
    CMAP0.set_under(CMAP0(0))  # set under-color to the lowest colormap color

    plt.ioff()
    
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    def get_array_and_unit(key):
        if z_from_t and key == 'z':
            return -299792458.0 * particle_group['delta_t'], 'm'
        return particle_group[key], particle_group.units(key).unitSymbol

    # Scale to nice units and get the factor, unit prefix
    x, u1 = get_array_and_unit(key1)
    y, u2 = get_array_and_unit(key2)
    
    # Form nice arrays
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(y, nice=nice, lim=ylim)
    
    w = particle_group['weight']
    
    ux = p1+u1
    uy = p2+u2
    
    labelx = pmd_beamphysics.labels.mathlabel(key1, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(key2, units=uy, tex=tex)

    if embedded:
        gs = GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[i], wspace=0.0, hspace=0.0)
    else:
        fig = plt.figure(**kwargs)
        gs = GridSpec(4,4)
    
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4,3], sharey=ax_joint)

    # Set the joint plot background color to match the bottom end of the colormap
    ax_joint.set_facecolor(CMAP0(0) if background is None else background)
    
    # Plot the hexbin
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)
    
    # Top histogram
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    if u1 == 's':
        _, hist_prefix = pmd_beamphysics.units.nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:   
        ax_marg_x.set_ylabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{ux}'))

    # Side histogram
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{uy}'))
    
    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)
    
    if xlim:
        ax_joint.set_xlim(xmin/f1, xmax/f1)      
        ax_marg_x.set_xlim(xmin/f1, xmax/f1)
        
    if ylim:
        ax_joint.set_ylim(ymin/f2, ymax/f2)     
        ax_marg_y.set_ylim(ymin/f2, ymax/f2)

    # Restore display only when drawing into a caller's figure; otherwise every later figure
    # in the session would silently stop being shown.
    if embedded and was_interactive:
        plt.ion()
    
    return fig


 



def slicePlotMod(particle_group, 
               *keys,
               n_slice=40,
               slice_key=None,
               xlim=None,
               ylim=None,
               tex=True,
               nice=True,
               **kwargs):
    """
    Create a slice plot of beam statistics for a ParticleGroup, derived from openPMD-beamphysics slice_plot().

    Parameters
    ----------
    particle_group : ParticleGroup
        The particle group to plot.
    *keys : str
        Keys for the y-axis variables to plot (e.g., 'mean_px', 'mean_py').
    n_slice : int, default 40
        Number of slices to divide the beam into along the slice_key.
    slice_key : str, optional
        Key to slice along (e.g., 'z' or 't'). If None, uses 'z' if in t-coordinates, else 't'.
    xlim : tuple, optional
        Limits for the x-axis.
    ylim : tuple, optional
        Limits for the y-axis.
    tex : bool, default True
        Whether to use TeX-style labels.
    nice : bool, default True
        Whether to use nice units and scaling for axes.
    **kwargs : dict
        Additional keyword arguments passed to plt.subplots().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.

    Notes
    -----
    - Plots the mean value of each key in each slice, with density shown as a filled region on a secondary y-axis.
    - Axes are labeled with units and optionally TeX formatting.
    - Useful for visualizing longitudinal structure and correlations in the beam.
    """    

    #NMM new
    plt.close('all')
    
    #NMM new!
    plt.ioff()

    
    # Allow a single key
    #if isinstance(keys, str):
    # 
    #     keys = (keys, )
    
    if slice_key is None:
        if particle_group.in_t_coordinates:
            slice_key = 'z'
        else:
            slice_key = 't'  
            
    # Special case for delta_
    if slice_key.startswith('delta_'):
        slice_key = slice_key[6:]
        has_delta_prefix = True
    else:
        has_delta_prefix = False
    
    # Get all data
    x_key = 'mean_'+slice_key
    slice_dat = particle_group.slice_statistics(*keys, n_slice=n_slice, slice_key=slice_key)
    slice_dat['density'] = slice_dat['charge']/ slice_dat['ptp_'+slice_key]
    y2_key = 'density'

    # X-axis
    x = slice_dat['mean_'+slice_key]  
    if has_delta_prefix:
        x -= particle_group['mean_'+slice_key]
        slice_key = 'delta_'+slice_key # restore        
        
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    ux = p1+str(particle_group.units(slice_key))
    
    # Y-axis
    
    # Units check
    ulist = [particle_group.units(k).unitSymbol for k in keys]
    uy = ulist[0]
    if not all([u==uy for u in ulist] ):
        raise ValueError(f'Incompatible units: {ulist}')
    
    ymin = max([slice_dat[k].min() for k in keys])
    ymax = max([slice_dat[k].max() for k in keys])
    
    _, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(np.array([ymin, ymax]), nice=nice, lim=ylim)
    uy = p2 + uy
        
    # Form Figure
    fig, ax = plt.subplots(**kwargs)
    
    # Main curves  
    if len(keys) == 1:
        color = 'black'
    else:
        color = None
    
    for k in keys:
        label = pmd_beamphysics.labels.mathlabel(k, units=uy, tex=tex)
        ax.plot(x, slice_dat[k]/f2, label=label, color=color)
    if len(keys) > 1:
        ax.legend()      

    # Density on r.h.s
    y2, _, prey2, _, _ = pmd_beamphysics.units.plottable_array(slice_dat[y2_key], nice=nice, lim=None)
    
    # Convert to Amps if possible
    y2_units = f'C/{particle_group.units(x_key)}'
    if y2_units == 'C/s':
        y2_units = 'A'
    y2_units = prey2+y2_units 
    
    # Labels
    labelx = pmd_beamphysics.labels.mathlabel(slice_key, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(*keys, units=uy, tex=tex)    
    labely2 = pmd_beamphysics.labels.mathlabel(y2_key, units=y2_units, tex=tex)        
    
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)


    # rhs plot
    ax2 = ax.twinx()
    ax2.set_ylabel(labely2)
    ax2.fill_between(x, 0, y2, color='black', alpha = 0.2)  
    ax2.set_ylim(0, None)
    
    # Actual plot limits, considering scaling
    if xlim:
        ax.set_xlim( xmin/f1, xmax/f1) 
    if ylim:
        ax.set_ylim( ymin/f2, ymax/f2)              

    return fig
