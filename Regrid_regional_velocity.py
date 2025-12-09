###############################################################################
# Notebook Details:
#.   Goal is to compute depth-integrated velocity from MOM6 NEP output
#.   Depth integrate u and v velocity components 
#.   Regrid velocity components to grid of plankton fields using geo-points
#.   Save u,v with geo-grid
###############################################################################
###############################################################################
#Import Packages
###############################################################################
# Add subroutine directory
import os
import sys
sys.path.append('/server/hpc/lol_scratch/jacojase/python/subroutines/')

#data
import xarray as xr
from netCDF4 import Dataset as nc4
import xesmf as xe

#maths
import numpy as np

#plotting
import matplotlib.pyplot as plt

#Custom packages
import regrid_tools_xesmf as rg

###############################################################################
# Input variables
###############################################################################
# veloctiy variable name
var_list = ['uo', 'vo']

# depth range to average over
zrange = slice(0,100)

# debug flag
plot_check = False

###############################################################################
#Define directories & Files: 5km Daily MOM6-NEP Model Output
###############################################################################
# Define root directory containing files
root = ''

# grid for mapping
gridpath = ''

# Define save directory
save_dir = ''

###############################################################################
#Local Subroutines
###############################################################################
def make_regridder_vel(filepath, gridpath, target_grid, point) : 
    """
    Generate Regridder object for u-point variables using 
    linear interpolation
    
    Input: filepath (string) : path to dataset.nc
           gridpath (string) : path to grid file static.nc
           target_grid (xarray) : lats and lons of target grid
           point (string) : either 'u' or 'v'
           
    Returns: regridder (xesmf regridder object)
    """
    # Open data file
    ds = xr.open_dataset(filepath, decode_timedelta=False)
    
    # Open grid file
    gf = xr.open_dataset(gridpath, decode_timedelta=False)
    
    # Read geographic lat and lon from grid file
    if (point == 'u') :
        lat = gf['geolat_u']
        lon = gf['geolon_u']
        
    elif (point == 'v') :
        lat = gf['geolat_v']
        lon = gf['geolon_v']
        
    else :
        print('Input "point" not "u" or "v", please change')
        return
    
    # Define lat and lon in data file
    ds['lon'] = lon
    ds['lat'] = lat
    
    # Generate regridder object
    regridder = xe.Regridder(ds, target_grid, 'bilinear')
    
    return regridder

###############################################################################
#Load Files: 5km Daily MOM6-NEP Model Output - Generate list of filenames
###############################################################################
##  Flag to set 2d or 3d fields
fileID = '_3d.nc'
saveID = '_5k_regrid.nc'

## Find files that match fileID store as list
filenames = rg.find_files(root, fileID)

# Sort files
filenames_sorted = sorted(filenames, key = lambda x: x[:8])

# Add full path to each file
file_paths = [os.path.join(root, fname) for fname in filenames_sorted]

###############################################################################
#Computation: Define target grid, store as xarray dataset
###############################################################################
# read lats and lons of geo locations at rho points
locs = {}
locs = rg.lats_lons(gridpath)

# define regular grid
lolim = (np.round(np.nanmin(locs['lon']), decimals = 2),
         np.round(np.nanmax(locs['lon']), decimals = 2),
         470)
lalim = (np.round(np.nanmin(locs['lat']), decimals = 2),
         np.round(np.nanmax(locs['lat']), decimals = 2),
         470)
lons = np.array(np.linspace(lolim[0], lolim[1], lolim[2]))
lats = np.array(np.linspace(lalim[0], lalim[1], lalim[2]))

# Save as xarray dataset
target_grid = xr.Dataset({'lat': (['lat'], lats),
                          'lon': (['lon'], lons)})

###############################################################################
#Computation: Create regridder objects
###############################################################################
# u-point regridder object
u_regridder = make_regridder_vel(file_paths[0], gridpath, target_grid, 'u')

# v-point regridder object
v_regridder = make_regridder_vel(file_paths[0], gridpath, target_grid, 'v')

###############################################################################
#Computation: Perform regridding on each field
###############################################################################
# Create dictionary to store data
all_u = []
all_v = []

# Loop over ncfiles in directory
for fname in file_paths:
    print('Regridding file: ' + fname[-31:])
    
    # Open Dataset
    ds = xr.open_dataset(fname, decode_timedelta = True)
    
    # Subset by depth & depth average
    ds = ds.sel(z_l = zrange)
    ds_bar = ds.mean(dim = 'z_l')
    
    # Regrid u-velocity
    u_re = u_regridder(ds_bar[var_list[0]])
    
    # Convert to data array
    da_u = xr.DataArray(u_re,
                    dims = (ds_bar[var_list[0]].dims[0],
                           "lat",
                           "lon"),
                    coords = {ds_bar[var_list[0]].dims[0] : ds_bar[var_list[0]].coords['time'],
                           "lat" : target_grid['lat'].values,
                           "lon" : target_grid['lon'].values})
    all_u.append(da_u)

    # Regrid v-velocity
    v_re = v_regridder(ds_bar[var_list[1]])
    
    # Convert to data array
    da_v = xr.DataArray(v_re,
                    dims = (ds_bar[var_list[1]].dims[0],
                           "lat",
                           "lon"),
                    coords = {ds_bar[var_list[1]].dims[0] : ds_bar[var_list[1]].coords['time'],
                           "lat" : target_grid['lat'].values,
                           "lon" : target_grid['lon'].values})
    all_v.append(da_v)

# Concatenate along time axis
# Concatenate all files along the time dimension
u_full = xr.concat(all_u, dim="time")
v_full = xr.concat(all_v, dim="time")

# Combine into one dataset
velocity_ds = xr.Dataset({"u": u_full, "v": v_full})
    
# Save to NetCDF
velocity_ds.to_netcdf(save_dir+"zAvg100m_velocity"+saveID)
    
