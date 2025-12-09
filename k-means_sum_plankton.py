###############################################################################
#Import Packages
###############################################################################
# Add subroutine directory
import os
import sys
sys.path.append('/server/hpc/lol_scratch/jacojase/python/subroutines/')

#data
from netCDF4 import Dataset as nc4
import xarray as xr

#maths
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#plotting
import matplotlib.pyplot as plt

###############################################################################
#Input parameters
###############################################################################
# Variable name
#varnames = ['nlgz_100', 'nmdz_100', 'nsmz_100',
#            'nlgp_100', 'nmdp_100', 'nsmp_100']

varnames = ['nlgz_100', 'nmdz_100', 'nsmz_100']

# Output file name
outdir = '/server/hpc/lol_scratch/jacojase/data/'
filename = 'FullDomain_SumZooplankton_ClusterMaps_min03.nc'


# Range of clusters numbers
min_cluster = 3
max_cluster = 9


# Spatial subsetting based on xh yh grid
loc = {}
loc['x_min'] = 230
loc['x_max'] = 237
loc['y_min'] = 39
loc['y_max'] = 49


# Initialize dictionary to store clusters
cluster_maps = {}

# Initialize dictionary to store phenology/climatology
clim = {}

plot_check = False
print_diags = False

###############################################################################
#Local subroutines
###############################################################################
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

def make_selector_list(varnames) :
    """
    Create function to select a single variable to open.
    This is to be used with xarray.open_mfdataset.
    
    Input: varname (string) : variable name
    
    Returns: select_var (function) : selects variable 
    """
    if isinstance(varnames, str):
        varnames = [varnames]

    def select_var(ds):
        return ds[varnames]
    
    return select_var


def annual_phenology(file_paths, varnames, loc, clim) :
    """
    Compute day-of-year phenology at each grid point
    
    Input: file_paths (list)    : list of NetCDF files to open
           varnames (list)      : list of variable names to load
           loc (dict)           : dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
           clim (dict)          : dictionary to store climatology fields

    Returns:
           clim (dict)          : contains 3D (dayofyear, lat, lon) maps of:
                                  budget terms and residual
    """
    # Construct variable list
    select_fun = make_selector_list(varnames)
    
    # Open dataset
    ds = xr.open_mfdataset(file_paths,
                           engine='netcdf4',
                           combine='by_coords',
                           preprocess=select_fun,
                           decode_timedelta=False)
    
    # Spatial Subsetting
    regional_ds = ds.sel(lat=slice(loc['y_min'], loc['y_max']),
                         lon=slice(loc['x_min'], loc['x_max']))
    
    # Sum each variable in varnames
    summed = sum(regional_ds[v] for v in varnames)
    
    # Group by day of year
    grouped = summed.groupby("time.dayofyear")
            
    # Compute mean and store in clim dropping dayofyear = 366 
    clim = grouped.mean("time").sel(dayofyear=slice(1, 365))
    
    # Convert to Dataset
    clim = clim.to_dataset(name="summed_var")
    
    return clim

###############################################################################
#Directories & Files: 5km Daily MOM6-NEP Model Output
###############################################################################
root = ''

# List files
filenames = find_files(root, '_2d.nc')

# Sort files
filenames_sorted = sorted(filenames, key = lambda x: x[:8])

# Add full path to each file
file_paths = [os.path.join(root, fname) for fname in filenames_sorted]

###############################################################################
#Computation: Annual Phenology 
###############################################################################
# Phenology
var_ds = annual_phenology(file_paths, varnames, loc, clim)

# Enforce spatial subsetting
var_ds['summed_var'] = var_ds['summed_var'].sel(lat=slice(loc['y_min'], 
                                                               loc['y_max']),
                                                lon=slice(loc['x_min'], 
                                                               loc['x_max']))

###############################################################################
#Computation: K-means clustering
###############################################################################
# Assign variable for processing
if (print_diags == True) :
    print(f"Processing variable {var}")
# Read variabels
da = var_ds['summed_var']
    
#clean (remove) mask
da = da.where(da != 0)
da = da.where(~np.isnan(da), drop = False)
    
#reshape to [space, time]
da_flat = da.stack(space=('lat', 'lon')).transpose('space', 'dayofyear')
    
#clean reshaped data (find non-masked spatial points)
valid_mask = ~np.isnan(da_flat).any(dim='dayofyear')
valid_mask_np = valid_mask.compute().values
    
# select valid points
da_valid = da_flat.isel(space=np.where(valid_mask_np)[0])
da_valid = da_valid.compute()
    
#K-means
# normalize time series
scaler = StandardScaler()
x_scaled = scaler.fit_transform(da_valid.values)
    
# Automatic cluster number detection using silhouette score
best_score = -1
best_n_clusters = min_cluster
for n in range(min_cluster, max_cluster):  
    kmeans_test = KMeans(n_clusters=n, random_state=0)
    labels_test = kmeans_test.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels_test)
    if score > best_score:
        best_score = score
        best_n_clusters = n
    
print(f"Optimal number of clusters: {best_n_clusters}")
print(f"Silhouette Score: {best_score}")
    
# apply clustering
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
labels = kmeans.fit_predict(x_scaled)
    
#Reshape clusters back to spatial map
# total number of spatial points
npts = da_flat.sizes['space']
    
# full list including places for land mask
labels_full = np.full(npts, np.nan)
labels_full[valid_mask.values] = labels
    
# reshape to (lat, lon)
labels_map = xr.DataArray(labels_full.reshape((da.sizes['lat'],
                                               da.sizes['lon'])),
                          coords={'lat': da['lat'], 
                                  'lon': da['lon']},
                          dims=('lat', 'lon'),
                          name="SumPlankton_clusters")

# store for clusters
cluster_maps = labels_map

###############################################################################
# Save cluster maps to NetCDF
###############################################################################
# Combine all variable maps into one Dataset
cluster_ds = xr.Dataset({cluster_maps.name: cluster_maps})

# Save to NetCDF (edit path as needed)
cluster_ds.to_netcdf(outdir+filename)
