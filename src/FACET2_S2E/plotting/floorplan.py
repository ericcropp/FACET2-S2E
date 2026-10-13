"""Beamline floorplan plots alongside longitudinal lattice functions.

Moved from UTILITY_plotMod.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import io
import pandas as pd


########
#The following floorplan plots and support functions are adapted from functions from D. Cesar
########

colorlist=['#982649', '#6BCAE2', '#72bda3', '#ed6a5a', '#533a71']


colorlist2=['#E2836A','#6a7ee2','#74e26a']


def floorplan_sorter(ele):
    """
    ele is an element of a pandas dictionary made from the bmad floorplan (made to look like an elegant floorplan from my "elegant_helpers" file). This sorter parses the columns to make a "patch" for plotting purposes. See floorplan_patches().
    Parse a floorplan element dictionary and return a matplotlib patch for plotting.

    Parameters
    ----------
    ele : dict or pandas.Series
        Element from a pandas DataFrame representing the floorplan.

    Returns
    -------
    patch : matplotlib.patches.Patch or None
        Patch object for plotting the element, or None if not applicable.

    Notes
    -----
    - The patch color and type are determined by the element's type and name.
    - Used internally by floorplan_patches().
    
    """
    if ele['ds']==0:
        ele['ds']=0.05
    s0=float(ele['s'])-float(ele['ds'])
    x=float(ele['X'])*2-0.5
    
    if (re.search('DRIF',ele.ElementType.upper())!=None)|(ele.ElementType.upper()=='MARK'):
        patchColor=None
        patch=None
    elif (re.search('BEND',ele.ElementType.upper())!=None)|(re.search('KICK',ele.ElementType.upper())!=None):
        patchColor='r'
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif re.search('XL',ele.ElementName.upper())!=None:
        patchColor=colorlist2[1]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'QUAD' in ele.ElementType.upper():
        patchColor=colorlist[2]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'SEXT' in ele.ElementType.upper():
        patchColor="#204337"
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    elif 'RFCW' in ele.ElementType.upper() or 'CAV' in ele.ElementType.upper():
        string_lst=['L3_10_50','L3_10_25','L2_10_25','L2_10_50','X1_Xband','L1_10_25','L1_9_25','L1_9_50'] #nonzero voltage
        if re.findall(r"(?=("+'|'.join(string_lst)+r"))",ele.ElementName)!=None:
            patchColor="#CD7F32"
            patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=0.5,color=patchColor,lw=0,ls=None)
    elif re.search('^UM',ele.ElementName)!=None:
        patchColor=colorlist[1]
        patch=mpatches.Rectangle(xy=(s0,x),width=float(ele['ds']),height=1,color=patchColor,lw=0,ls=None)
    else:
        patchColor=None
        patch=None
        
    return patch


def floorplan_patches(floorplan,zbounds=None):
    """
    This function returns a list of patches to be plotted (patches) and a list of patches for the legend (leg_patches). If zbounds=[zmin,zmax] is given then the plot is restricted to the bounds. 
    
    Usage:
    
    fp=SDDS(0)
    fp.load(LCLS2scS.flr)
    __,floorplan=sdds2pd(fp)
    patches,leg_patches=floorplan_patches(floorplan,[3425,3750])

    Parameters
    ----------
    floorplan : pandas.DataFrame
        DataFrame containing floorplan information (columns: s, ds, ElementType, etc.).
    zbounds : list or tuple, optional
        [zmin, zmax] bounds for restricting the plot range. If None, uses full range.

    Returns
    -------
    patches : list
        List of matplotlib patch objects for the floorplan.
    leg_patches : list
        List of patch objects for the legend.

    Notes
    -----
    - Used for visualizing the lattice layout in longitudinal plots.
    """
    if zbounds==None:
        zbounds=[floorplan['s'].min(),floorplan['s'].max()]
    sFloor=floorplan.s.astype(dtype=float);
    sFloor=sFloor.values
    ii=0;
    patches=[];
    for index, ele in (floorplan.iloc[(sFloor>zbounds[0])&(sFloor<zbounds[1])]).iterrows():
        patches.append(floorplan_sorter(ele))
    
    quad_patch = mpatches.Patch(color=colorlist[2], label='Quad')
    sext_patch = mpatches.Patch(color="#204337", label='Linac')
    bend_patch = mpatches.Patch(color='red', label='Bend')
    leg_patches=[quad_patch,sext_patch,bend_patch];
    return patches,leg_patches


def floorplan_plot_partial(ax_fp,floorplan,zmin=0,zmax=2000):  
    """
    This function plots "patches" for basic elements in the lattice. This can help identify what you're looking at in a "z-plot".
    Parameters
    ----------
    ax_fp : matplotlib.axes.Axes
        The axis to plot the floorplan patches on.
    floorplan : pandas.DataFrame
        DataFrame containing floorplan information.
    zmin : float, default 0
        Minimum z position for the plot.
    zmax : float, default 2000
        Maximum z position for the plot.

    Returns
    -------
    ax_fp : matplotlib.axes.Axes
        The axis with the floorplan patches added.

    
    """
    
    patches,leg_patches=floorplan_patches(floorplan,[zmin,zmax])

    for p in patches:
        if p!=None:
            ax_fp.add_patch(p)

    ax_fp.plot((zmin,zmax),(0,0),'k',alpha=0.0)
    ax_fp.tick_params(axis='x',direction='out',length=15,width=6,color='k',bottom=True)
    plt.yticks([])
    #ax_fp.set_ylim([-3,1])
    ax_fp.set_xlim([zmin,zmax])
    return ax_fp


    
def format_longitudinal_plot(fig, floorplan):
    """
    This function helps format a "z-plot" by providing axes for the main plot and for the a floorplan_plot_partial. It also plots the floorplan.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add axes to.
    floorplan : pandas.DataFrame
        DataFrame containing floorplan information.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The main plot axis.
    ax_fp : matplotlib.axes.Axes
        The floorplan axis.

    
    """
    outer_grid=fig.add_gridspec(5,1,hspace=0)
    ax=fig.add_subplot(outer_grid[0:4,:])
    ax_fp=fig.add_subplot(outer_grid[4,:], sharex = ax)
    floorplan_plot_partial(ax_fp, floorplan)
    plt.sca(ax)
    
    return ax, ax_fp 


def floorplanPlot(
    tao,
    zmin = 13,
    zmax = 1020,
    ymin = 0.1,
    ymax = 250
):
    """
    Plot the lattice floorplan and key beamline functions (beta, eta, etc.)

    Parameters
    ----------
    tao : Tao
        Active Tao simulation object.
    zmin : float, default 13
        Minimum z position for the plot.
    zmax : float, default 1020
        Maximum z position for the plot.
    ymin : float, default 0.1
        Minimum y-axis value for beta plots.
    ymax : float, default 250
        Maximum y-axis value for beta plots.

    Returns
    -------
    None

    Notes
    -----
    - Reads the lattice and floorplan from Tao, extracts beta and eta functions, and plots them with the floorplan.
    - Uses logarithmic scaling for beta functions and overlays eta functions on a secondary y-axis.
    - Intended for visualizing the beamline layout and optics.
    """

    elements=tao.lat_ele_list();

    
    floorplan=pd.read_csv(
        io.StringIO('\n'.join(tao.show('lat -all -floor_coords -tracking_elements')[3:-5])), 
        sep=r"[\s\n]+",
        engine='python',
        names=['Index','ElementName','ElementType','s','ds','X','Y','Z','Theta','Phi','Psi'])
    floorplan.drop(0,inplace=True)
    
    #Get twiss functions
    #NMM 2025-02-04: I'm swapping out these x/y for a/b for eta and beta. I think they get confused later so just removing any and all ambiguity
    tao.cmd('set global lattice_calc_on = T')
    s=np.array([tao.lat_list(x,'ele.s')[0] for x in floorplan.Index])
    x=np.array([tao.lat_list(x,'orbit.floor.x')[0] for x in floorplan.Index])
    beta_a=np.array([tao.lat_list(x,'ele.a.beta')[0] for x in floorplan.Index])
    beta_b=np.array([tao.lat_list(x,'ele.b.beta')[0] for x in floorplan.Index])    
    # beta_y=np.array([tao.lat_list(x,'ele.a.beta')[0] for x in floorplan.Index])
    # beta_x=np.array([tao.lat_list(x,'ele.b.beta')[0] for x in floorplan.Index])
    etot=np.array([tao.lat_list(x,'ele.e_tot')[0] for x in floorplan.Index])

    #NMM 2025-02-04: I'm swapping out these eta_x/y for a/b. I think they get confused later so just removing any and all ambiguity
    #eta_y=np.array([tao.lat_list(x,'ele.y.eta')[0] for x in floorplan.Index])
    #eta_x=np.array([tao.lat_list(x,'ele.x.eta')[0] for x in floorplan.Index])

    eta_b=np.array([tao.lat_list(x,'ele.b.eta')[0] for x in floorplan.Index])
    eta_a=np.array([tao.lat_list(x,'ele.a.eta')[0] for x in floorplan.Index])
    
    fig = plt.figure(num=1,figsize=[3.375*5,3.375*2])
    fig.clf()
    ax,ax_fp=format_longitudinal_plot(fig, floorplan)

    ax.semilogy(s,beta_a,label='beta a')
    ax.semilogy(s,beta_b,label='beta b')
    
    
    

    # ax.semilogy(s,beta_x,label='beta b')
    # ax.semilogy(s,beta_y,label='beta a')
    plt.legend(loc=2)
    ax.set_ylim([ymin, ymax])
    ax_r=ax.twinx()
    #ax_r.plot(s,eta_x*1e3,'C0--',label='eta b')
    #ax_r.plot(s,eta_y*1e3,'C1--',label='eta a')
    ax_r.plot(s,eta_a*1e3,'C0--',label='eta a')
    ax_r.plot(s,eta_b*1e3,'C1--',label='eta b')
    
    
    
    plt.legend(loc=1)
    
    
    ax.set_facecolor('w')
    
    ax.set_xlabel('Z [m]',fontsize=14)
    ax.set_ylabel(r'$\beta$ [m]',fontsize=14)
    ax_r.set_ylabel(r'$\eta$ [mm]',fontsize=14)
    
    ax.set_xlim([zmin,zmax])
    ax_fp.set_ylim([-1,3])
    
    plt.show()

    #fig.savefig('beamline',transparent=False,bbox_inches='tight', dpi=300)
