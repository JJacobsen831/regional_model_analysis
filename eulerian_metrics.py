###############################################################################
#Import Packages
###############################################################################
# Add subroutine directory
import os
import sys
sys.path.append('../subroutines/')

#data
import xarray as xr

#math
import numpy as np

#plotting
import matplotlib.pyplot as plt
import cmocean 
import cartopy
import cartopy.feature as cfeature

###############################################################################
#Input parameters: 
###############################################################################
# files
root = ''
ncid = '~.nc'

# variable names
varlist = ['uo', 'vo']

# figure title
title = 'Jan 2022 - Dec 2023'

###############################################################################
#Local subroutines
###############################################################################
def make_selector_list(varnames) :
    """
    Create function to select a multiple variables to open.
    This is to be used with xarray.open_mfdataset.
    
    Input: varname (list) : strings of variable names
    
    Returns: select_var (function) : selects variable 
    """
    if isinstance(varnames, str):
        varnames = [varnames]

    def select_var(ds):
        return ds[varnames]
    
    return select_var

def geo_to_meters(ds) :
    # read lat/lon
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # reference point
    lat0 = np.mean(lat[0])
    lon0 = np.mean(lon[0])

    # Constants
    R = 6371000.0  # mean Earth radius (m)
    deg_to_rad = np.pi / 180.0
    
    # convert to radians
    dlat = (lat - lat0) * deg_to_rad
    dlon = (lon - lon0) * deg_to_rad
    
    # convert to meters
    x = R * dlon * np.cos(lat0 * deg_to_rad)
    y = R * dlat
    
    #save as new dataset with meters as dimensions
    ds_out = ds.copy()
    ds_out = ds_out.assign_coords(x=(ds['lon'].dims, x),
                                  y=(ds['lat'].dims, y))

    return ds_out

def eulerian_metrics(ds_in):
    """
    Compute divergence, vorticity, strain, and Okubo-Weiss parameter 
    from Eulerian velocities
    
    Input: ds (xarray dataset) : velocity fields
    
    Returns ds (xarray dataset) : eulerian metrics
    
    """
    #convert grid to meters
    ds = geo_to_meters(ds_in)
    
    # Read velocity fields
    u = ds['u']  # dims: time, y, x
    v = ds['v']
    
    # compute gradients 
    du_dx = u.differentiate('x')  
    du_dy = u.differentiate('y')
    dv_dx = v.differentiate('x')
    dv_dy = v.differentiate('y')

    # divergence
    div = du_dx + dv_dy

    # vorticity (z-component)
    vort = dv_dx - du_dy

    # strain components
    s_n = du_dx - dv_dy # normal strain (Sxx - Syy)
    s_s = du_dy + dv_dx # shear-like term (without 1/2 scaling convention)
    
    # Okubo-Weiss parameter (common definition)
    OW = s_n**2 + s_s**2 - vort**2

    # Q-criterion (2D simplified) optionally:
    # Sxx = du_dx; Syy = dv_dy; Sxy = 0.5*(du_dy + dv_dx)
    # S_norm2 = 2*(Sxy**2) + (Sxx**2 + Syy**2 - Sxx*Syy)  # more work; OW is fine for 2D

    return xr.Dataset({
        'div': div,
        'vort': vort,
        's_n': s_n,
        's_s': s_s,
        'okubo_weiss': OW
    })

def extract_mask(ds) :
    var = ds['uo'].values
    mask = var == np.nan

###############################################################################
#Read variables: Grid
###############################################################################
# Grid rooot & file
gridid = '/server/hpc/lol_scratch/FlowFields/mom6nep/NEP5k_newport_ocean_static.nc'
grid = xr.open_dataset(gridid, 
                       decode_timedelta = False)

# Location subsetting [Do not change]
loc = {}
loc['time'] = slice('2022-01-01', '2023-12-31')
loc['lat_min'] = 39
loc['lat_max'] = 49
loc['lon_min'] = 230
loc['lon_max'] = 237

###############################################################################
#Directories & Files: Open dataset 
###############################################################################
# Open combined dataset
ds = xr.open_dataset(root+ncid,
                       decode_timedelta = False)

ds = ds.sel(time=loc['time'],
            lat=slice(loc['lat_min'], loc['lat_max']),
            lon=slice(loc['lon_min'], loc['lon_max']))
                 

mask = np.isnan(ds['u'][0,:,:].values)

###############################################################################
#Computation: Eularian metrics & strain locations
###############################################################################
# div, vort, strain (okubo_weiss)
emet = eulerian_metrics(ds)

# locate regions of high strain (OW > 90th ptile)
ow = emet['okubo_weiss']
ow_pcts = np.nanpercentile(ow.values, [75, 90], axis=0)

#. Threshold of 90th percentile
ow_thresh = float(np.nanpercentile(ow.values, 90))

#. Fraction of days exceeding threshold
ow_bool = ow > ow_thresh

#. Total exceedence
ow_exceed = ow_bool.mean(dim='time')

#. Seasonal exeedence
ow_seas = ow_bool.groupby('time.season').mean(dim = 'time')

###############################################################################
#Computation: Convergent locations
###############################################################################
# locate regions of convergence
div = emet['div']
div_pcts = np.nanpercentile(div.values, [10, 25], axis=0)

#. Treshold 10 percentile
div_thresh = float(np.nanpercentile(div.values, 10))

#. Fraction of days exceeding threshold (conv is negative so less than 10%)
div_bool = div < div_thresh

# Total exceedence
div_exceed = div_bool.mean(dim='time')

# Seasonal exceedence
div_seas = div_bool.groupby('time.season').mean(dim = 'time')

###############################################################################
#Computation: Apply Mask
###############################################################################
div_exceed = div_exceed.where(~mask)
div_seas = div_seas.where(~mask)
ow_exceed = ow_exceed.where(~mask)
ow_seas = ow_seas.where(~mask)

###############################################################################
#Plotting: Plot Parameters
###############################################################################
cpar = (0, 1, 0.1)
levs = np.arange(cpar[0], cpar[1]+cpar[2], cpar[2])

cmap = cmocean.cm.dense

###############################################################################
#Plotting: Figure showing total exceedence
###############################################################################
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (9,9),
                               sharex = True, sharey = True,
                               subplot_kw={"projection": cartopy.crs.PlateCarree()})
# Proportion of days with high strain
cf1 = ax1.contourf(ow_exceed['lon']-360, ow_exceed['lat'], ow_exceed.values,
                   vmin = cpar[0], vmax = cpar[1],
                   levels = levs,
                   cmap = cmap,
                   transform=cartopy.crs.PlateCarree(),
                   zorder = 1)

# bathymetry
ba1 = ax1.contour(grid['geolon']-360, grid['geolat'],
                  grid['deptho'],
                  levels = [200],
                  colors = 'w',
                  alpha = 0.5,
                  linewiths = 0.1,
                  #linestyles = 'dashed',
                  transform=cartopy.crs.PlateCarree(),
                  zorder = 4)
ax1.clabel(ba1, inline=True, fontsize=5)

ax1.set_title(title, loc = 'left')

# Add coastlines
ax1.coastlines('10m')
ax1.set_extent([loc['lon_min'], loc['lon_max'], 
               loc['lat_min'], loc['lat_max']],
               crs=cartopy.crs.PlateCarree())
ax1.set_xticks(np.arange(loc['lon_min']-360, loc['lon_max']-360,1),
               crs=cartopy.crs.PlateCarree())

ax1.tick_params(axis = 'y', label1On=True)
ax1.set_yticks(np.arange(loc['lat_min'], loc['lat_max'],1),
               crs=cartopy.crs.PlateCarree())

# Add labels 
ax1.set_ylabel('Latitude')
ax1.tick_params(axis='x', labelrotation=15) 
ax1.set_aspect('equal')
ax1.patch.set_facecolor('silver')
ax1.grid()
fig.colorbar(cf1, label = 'Proportion of Days \n with High Strain')

# Proportion of days with high convergence
cf2 = ax2.contourf(div_exceed['lon']-360, div_exceed['lat'], div_exceed.values,
                   vmin = cpar[0], vmax = cpar[1],
                   levels = levs,
                   cmap = cmap,
                   transform=cartopy.crs.PlateCarree(),
                   zorder = 1)

# bathymetry
ba2 = ax2.contour(grid['geolon']-360, grid['geolat'],
                  grid['deptho'],
                  levels = [200],
                  colors = 'w',
                  alpha = 0.5,
                  linewiths = 0.1,
                  #linestyles = 'dashed',
                  transform=cartopy.crs.PlateCarree(),
                  zorder = 4)
ax2.clabel(ba2, inline=True, fontsize=5)

# coastlines
ax2.coastlines('10m')
ax2.set_extent([loc['lon_min'], loc['lon_max'], 
                loc['lat_min'], loc['lat_max']],
               crs=cartopy.crs.PlateCarree())

ax2.tick_params(axis = 'x', label1On=True)
ax2.set_xticks(np.arange(loc['lon_min']-360, loc['lon_max']-360,1),
               crs=cartopy.crs.PlateCarree())
     
ax2.tick_params(axis = 'y', label1On=True)
ax2.set_yticks(np.arange(loc['lat_min'], loc['lat_max'],1),
               crs=cartopy.crs.PlateCarree())

# Add labels 
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.tick_params(axis='x', labelrotation=15) 
ax2.set_aspect('equal')
ax2.patch.set_facecolor('silver')
ax2.grid()
fig.colorbar(cf2, label = 'Proportion of Days \n with High Convergence')

###############################################################################
#Plotting: Figure showing seasonal exceedence for High Strain
###############################################################################
season_order = ['DJF', 'MAM', 'JJA', 'SON']
seasons = [s for s in season_order if s in ow_seas['season'].values]
ncols, nrows = 2,2
fig, axes = plt.subplots(nrows, ncols, figsize=(10,8),
                         subplot_kw={"projection": cartopy.crs.PlateCarree()})
axes = axes.flatten()
for i, season in enumerate(seasons):
    ax = axes[i]
    
    cf = ax.contourf(ow_seas['lon']-360, ow_seas['lat'],
                     ow_seas.sel(season=season).values,
                     vmin=cpar[0], vmax=cpar[1],
                     levels=levs,
                     cmap=cmap,
                     transform=cartopy.crs.PlateCarree(),
                     zorder=1)

    # Bathymetry
    ba = ax.contour(grid['geolon']-360, grid['geolat'],
                    grid['deptho'],
                    levels=[200],
                    colors='w',
                    alpha=0.8,
                    linewidths=1,
                    transform=cartopy.crs.PlateCarree(),
                    zorder=2)

    # Coastlines and extent
    ax.coastlines('10m')
    ax.set_extent([loc['lon_min'], loc['lon_max'], 
                   loc['lat_min'], loc['lat_max']], 
                  crs=cartopy.crs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='silver')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    
    # Ticks: only bottom row and left column
    gl.top_labels = False
    gl.right_labels = False
    if i % ncols != 0:  # hide left labels for right column
        gl.left_labels = False
    if i // ncols != nrows-1:  # hide bottom labels for top row
        gl.bottom_labels = False
    
    # Titles
    ax.set_title(season, loc='left')



# Add colorbar 
fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.95, wspace=0.000001, hspace=0.1)

# Add colorbar to the right
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical', label='Proportion of Days\nwith High Strain')

###############################################################################
#Plotting: Figure showing seasonal exceedence for High Convergence
###############################################################################
season_order = ['DJF', 'MAM', 'JJA', 'SON']
seasons = [s for s in season_order if s in ow_seas['season'].values]
ncols, nrows = 2,2
fig, axes = plt.subplots(nrows, ncols, figsize=(10,8),
                         subplot_kw={"projection": cartopy.crs.PlateCarree()})
axes = axes.flatten()
for i, season in enumerate(seasons):
    ax = axes[i]
    
    cf = ax.contourf(div_seas['lon']-360, div_seas['lat'],
                     div_seas.sel(season=season).values,
                     vmin=cpar[0], vmax=cpar[1],
                     levels=levs,
                     cmap=cmap,
                     transform=cartopy.crs.PlateCarree(),
                     zorder=1)

    # Bathymetry
    ba = ax.contour(grid['geolon']-360, grid['geolat'],
                    grid['deptho'],
                    levels=[200],
                    colors='w',
                    alpha=0.8,
                    linewidths=1,
                    transform=cartopy.crs.PlateCarree(),
                    zorder=4)

    # Coastlines and extent
    ax.coastlines('10m')
    ax.set_extent([loc['lon_min'], loc['lon_max'], 
                   loc['lat_min'], loc['lat_max']], 
                  crs=cartopy.crs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='silver')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    
    # Ticks: only bottom row and left column
    gl.top_labels = False
    gl.right_labels = False
    if i % ncols != 0:  # hide left labels for right column
        gl.left_labels = False
    if i // ncols != nrows-1:  # hide bottom labels for top row
        gl.bottom_labels = False
    
    # Titles
    ax.set_title(season, loc='left')



# Add colorbar 
fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.95, wspace=0.000001, hspace=0.1)

# Add colorbar to the right
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical', label='Proportion of Days\nwith High Convergence')

