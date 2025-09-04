from vtk.util.numpy_support import vtk_to_numpy
import vtk ,glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from sklearn import manifold

def natural_sort_key(s):
    """ Function for natural sorting (e.g. 'file2' < 'file10'). """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]
    
def extract_data_from_vts(vts_file, axis='z', index=None):
 
    # Check if file exists
    if not os.path.exists(vts_file):
        raise FileNotFoundError(f"VTS file not found: {vts_file}")
 
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(vts_file)
    reader.Update()
 
    # Check if reader encountered errors
    if reader.GetErrorCode() != 0:
        raise RuntimeError(f"VTK reader error code: {reader.GetErrorCode()}")
   
    grid = reader.GetOutput()
    dimensions = grid.GetDimensions()
    ny,nx,nz = dimensions
    
    # Check if grid is valid
    if grid is None:
        raise RuntimeError("Failed to read VTS file - grid is None")
   
    if grid.GetNumberOfPoints() == 0:
        raise RuntimeError("VTS file appears to be empty - no points found")
 
    # Only extract the data we need, not the full array
    data = np.reshape(vtk_to_numpy(grid.GetPointData().GetScalars()),(nz,nx,ny))

    #print('[nx,ny,nz]: '+str([nx,ny,nz]))
    return data#, dimensions

def plot_sections(array,ix=0,iy=0,iz=0,cmap='PuRd',title=None,label=None):
    vmin = np.nanmin(array.flatten())
    vmax = np.nanmax(array.flatten())
    slice_3_array = array[:,:,iy]
    slice_2_array = array[:,ix,:]
    slice_1_array = array[iz,:,:]
    # Define the height ratios
    # gs_kw = dict(height_ratios=[1, 6, 1])

    # Create subplots with the specified height ratios
    # fig, axs = plt.subplots(3,3,dpi=150,gridspec_kw=gs_kw, figsize=(10, 4))
    fig = plt.figure(dpi=150, figsize=(12, 4))
    gs = GridSpec(2, 3, figure=fig,height_ratios=[6,0.5])
    # ax0 = fig.add_subplot(gs[0, :])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[-1, :])

    fig.suptitle(title)
    im = ax1.imshow(slice_3_array,cmap=cmap,vmin=vmin,vmax=vmax)
    ax1.set_xlabel('x (px)')
    ax1.set_ylabel('z (px)')
    ax1.set_title('iy='+str(iy))
    ax2.imshow(slice_2_array,cmap=cmap,vmin=vmin,vmax=vmax)
    ax2.set_xlabel('y (px)')
    ax2.set_ylabel('z (px)')
    ax2.set_title('ix='+str(ix))
    ax3.imshow(slice_1_array,cmap=cmap,vmin=vmin,vmax=vmax)
    ax3.set_xlabel('y (px)')
    ax3.set_ylabel('x (px)')
    ax3.set_title('iz='+str(iz))
    fig.colorbar(im, cax=ax4, orientation='horizontal',label=label)
    # cbar_ax = fig.add_axes([0.12, 0.18, 0.78, 0.03]) # Adjust these values as needed
    # fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    plt.show()
    return

def plot_dissimilarity(distance_mx,title):
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1234,
                       dissimilarity="precomputed", n_jobs=1)
    
    mdspos = mds.fit(distance_mx).embedding_
    nmodels = len(distance_mx)
    s_id = np.arange(nmodels)
    # Plot concentric circle dataset
    colors1 = np.flipud(plt.cm.Blues(np.linspace(0., 1, 512)))
    colors2 = plt.cm.Greens(np.linspace(0, 1, 512))
    colors3 = np.flipud(plt.cm.Reds(np.linspace(0, 1, 512)))
    colors = np.vstack((colors1, colors2, colors3))
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    ix=np.tril_indices(nmodels,k=-1)
    
    lcmin = np.amin(distance_mx[ix]) 
    lcmax = np.amax(distance_mx[ix])
    
    lcMDSxmin = np.min(mdspos[:,0])
    lcMDSxmax = np.max(mdspos[:,0])
    lcMDSymin = np.min(mdspos[:,1])
    lcMDSymax = np.max(mdspos[:,1])
    
    s = 100
    
    fig,_ = plt.subplots(1,2,figsize=(12,4.5),dpi=300)
    plt.suptitle(title)
    plt.subplot(121)
    plt.imshow(distance_mx,cmap='inferno'),plt.colorbar()
    plt.xlabel('realization #')
    plt.ylabel('realization #')
    plt.title('dissimiarity matrix')
    plt.subplot(122)
    plt.title('2D Muti-Dimensional Scaling Representation')
    plt.scatter(mdspos[:, 0], mdspos[:, 1], c=s_id,cmap=mycmap, s=s, label='one realization', marker='+')
    plt.xlim(lcMDSxmin,lcMDSxmax)
    plt.ylim(lcMDSymin,lcMDSymax)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    cbar = plt.colorbar()
    cbar.set_label('sample #')
    plt.show()
    return fig