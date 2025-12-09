###############################################################################
# Notebook Details:
#.Goal is to compute correlation between MOM6 NEP output and NHL data
#.   Compute phenology from obs dataset
#.   Compute phenology from output for each zoop size class
#.   Spatial correlation of modeled zooplankton with NHL phenology
#.   Scatter plot with r2 with each size class
###############################################################################
###############################################################################
#Import Packages
###############################################################################
# Add subroutine directory
import os
import sys
sys.path.append('../subroutines/')

#data
from netCDF4 import Dataset as nc4
import xarray as xr
from datetime import datetime, timedelta
import calendar
import pandas as pd

#maths
import numpy as np

#plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

#custom packages
import mom6_tools as m6
import nhl_tools as nh

###############################################################################
#Input parameters
###############################################################################
# All zooplankton variables
varnames = ['nlgz_100', 'nmdz_100', 'nsmz_100']

# Select variable for analysis
title = 'Large_Zooplankton'
varname = varnames[0]

# station location
stn = (-124.6500519, 44.65175707)

# Spatial subsetting
locs = {}
locs['lat_min'] = 42
locs['lat_max'] = 46
locs['lon_min'] = 230
locs['lon_max'] = 237


#Save Directory
savedir = ''
savestr = savedir+title+'_Decorrelation.png'

###############################################################################
#Local Subroutines
###############################################################################
# Subroutines for model output
def make_selector(varname) :
    """
    Create function to select a single variable to open.
    This is to be used with xarray.open_mfdataset.
    
    Input: varname (string) : variable name
    
    Returns: select_var (function) : selects variable 
    """
    def select_var(ds) :
        return ds[[varname]]
    return select_var


def annual_clim(file_paths, varname, loc) :
    """
    Compute day-of-year climatology from model output
    
    Input: file_paths (list) : list of files to be concatenated
           varname (string)  : name of variable
           loc (dict)        : dictionary of lat and lon to interpolate to
           
    
    Returns: clim (ndarray) : dictionary of day-of-year climatology including
                           mean, 10% and 90% percentiles
    """
    # Function to select variable
    select_fun = make_selector(varname)
    
    # Open datasets
    ds = xr.open_mfdataset(file_paths, 
                           engine = 'netcdf4',
                           combine = 'by_coords',
                           preprocess = select_fun,
                           decode_timedelta = False)
    
    # Spatial subsetting for memory improvement
    da = ds.sel(lat=slice(loc['lat_min'], loc['lat_max']),
                lon=slice(loc['lon_min'], loc['lon_max']))
    
    # Group by day of year
    grouped = da.groupby("time.dayofyear")

    # Mean by day-of-year
    clim = grouped.mean("time").sel(dayofyear=slice(1,365))
    
    return clim


def find_files(root, string_id) :
    """
    Returns a list of files in the given directory that end with a specified 
    sequence.
    
    Input: root (string) : path to root directory
           string_id (string) : sequence at end of filename. 
                                Must inclide '.nc'
    
    Returns: list of file names that end with string_id

    """
    # define empty list
    matching_files = []
    
    # loop over files in directory
    for filename in os.listdir(root):
        # find files that end with string_id flag
        if filename.endswith(string_id):
            matching_files.append(filename)

    return matching_files

###############################################################################
#Directories & Files: Model output
###############################################################################
# Define root directory containing output files
root = '/server/hpc/lol_scratch/FlowFields/mom6nep/Regrid_NEP_5k/'

# Flag to set 2d or 3d fields
fileID = '_2d.nc'

## Find files that match fileID store as list
filenames = find_files(root, fileID)

# Sort files
filenames_sorted = sorted(filenames, key = lambda x: x[:8])

# Add full path to each file
file_paths = [os.path.join(root, fname) for fname in filenames_sorted]

###############################################################################
#Computation: Modeled Phenology averaged to monthly values 
###############################################################################
# Climatology: Daily
mod = {}
da = annual_clim(file_paths, varname, locs)

# Map dayofyear to month
months = pd.to_datetime(da['dayofyear'].values, format="%j").month

# Create a new DataArray for grouping
month_da = xr.DataArray(months, dims="dayofyear", 
                        coords={"dayofyear": da['dayofyear']})

# Group by month and average
mod['anom'] = da.groupby(month_da).mean(dim="dayofyear").to_array()

###############################################################################
#Computation: Subset Modeled Phenology to station location
###############################################################################
# Prep mod data for calculation
#. Drop 'variable' dimension in model output
mod_anom = mod['anom'].isel(variable=0)
mod_centered = mod_anom

mod_point = mod_anom.sel(lon = stn[0]+360,
                         lat = stn[1],
                         method = "nearest")

###############################################################################
#Computation: Correlation coefficient (r2)
###############################################################################
# Compute correlation along 'group' which is month
corr = xr.corr(mod_centered, mod_point, dim='group')

# Coeficient of determination
r_sq = corr**2

# Find maximum
#. Flatten Array
flat = r_sq.stack(points=('lat', 'lon'))

#. Convert to numpy arrays
flat_vals = flat.values  
lat_vals = flat['lat'].values
lon_vals = flat['lon'].values

#. Index of maximum
imax = np.nanargmax(flat_vals)

#. Extract values
max_val = flat_vals[imax]
max_lat = lat_vals[imax]
max_lon = lon_vals[imax]

#. Time series at max correlation
max_ts = mod_anom.sel(lat=max_lat, lon=max_lon)
ts_vals = max_ts.values

###############################################################################
#Plotting: Contourf plot of correlation with obs time series
#          Time series of obs and zoop at location of max correlation
#          Scatter plot of obs and zoop at location of max correlation
###############################################################################
# month names
months = [calendar.month_abbr[i] for i in range(1, 13)]

# Plotting Begins
fig = plt.figure(figsize=(12,6))

# ax1: Tall correlation map (left) with colorbar at bottom
ax1 = fig.add_axes([0.05, 0.05, 0.55, 0.9], projection=ccrs.Mercator())  # **tall map**
cf = ax1.contourf(mod_anom.lon-360, mod_anom.lat, corr**2,
                  vmin = 0, vmax = 1,
                  levels = np.arange(0, 1.1, 0.1),
                  transform=ccrs.PlateCarree(),
                  zorder=1, cmap= cmocean.cm.haline)
ax1.set_title(title, loc='left')
ax1.patch.set_facecolor('silver')
ax1.set_aspect('equal')
ax1.grid()
ax1.coastlines('10m')
ax1.set_extent([locs['lon_min'], locs['lon_max'], 
                locs['lat_min'], locs['lat_max']])

# Add NH25 location
ax1.scatter(-124.6500519, 44.65175707, c='red', marker='*',
            transform=ccrs.PlateCarree(), zorder=2)
ax1.text(-125.75, 44.7, 'NH25', color='red',
         transform=ccrs.PlateCarree(), zorder=3)

# Axis ticks
ax1.set_xticks(np.arange(locs['lon_min']-360, locs['lon_max']-360+1,1),
               crs=ccrs.PlateCarree())
ax1.set_xticklabels(np.arange(locs['lon_min']-360, locs['lon_max']+1-360,1))
ax1.set_xlabel('Lon [$^\circ$ E]')
ax1.set_yticks(np.arange(locs['lat_min'], locs['lat_max']+1,1),
               crs=ccrs.PlateCarree())
ax1.set_yticklabels(np.arange(locs['lat_min'], locs['lat_max']+1,1))
ax1.set_ylabel('Lat [$^\circ$ N]')

# Colorbar at bottom
cb = fig.colorbar(cf, ax=ax1, orientation='horizontal', 
                  fraction=0.025, pad=0.1, label=r"R$^{2}$")

# ax2: Time series (upper right)
ax2 = fig.add_axes([0.55, 0.55, 0.4, 0.3])  
ax2.plot(np.arange(1,13), mod_point, 
         color='k', linewidth=2, linestyle='-',
         marker='*', markerfacecolor='red', markeredgecolor = 'red')

ax2.set_xticks(np.arange(1,13))
ax2.set_xticklabels(months)
ax2.grid()
ax2.set_xlabel('Time [Month]')
ax2.set_ylabel('Biomass \n [mol m$^{-3}$]')
ax2.set_title('Time Series at NH25')

plt.savefig(savestr, dpi = 300)
