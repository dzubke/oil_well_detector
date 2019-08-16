# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'jupyter_notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Oil well detector

#%%
import rasterio as rio
from matplotlib import pyplot as plt
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show
from rasterio import features
from shapely.geometry import Point, box
import geopandas as gpd
from IPython.display import Image
import numpy as np

#%% [markdown]
# Detections oil wells in the Bakken oil field based on satellite imagery
# 
# For this project, I will be training an image recognition algorithm to recognize if a pixel is part of an oil well. I will be analyzing various types of satellite images to conduct this analysis.
# 
# An example of what these oil well look like is shown below (which for some reason I haven't yet figure out how to show the image in markdown text...)
# <img src='Non-images/notebook-images/Oil-well-example.png'>

#%%
Image(filename='non_images/notebook_images/Oil-well-example.jpg')

#%% [markdown]
# The image above shows three separate oil wells. 
#%% [markdown]
# ## Future Work
#%% [markdown]
# I am still working on this project. So far I have:
# 
#  - downloaded the relevant satelittle images (the raw images are too big to be in a github repo)
#  - utilized Rasterio and Shapely analyze the features of the image and crop an area of interest
# 
# Still to come I need to:
#  - Create the training data
#      - make outline examples of oil wells in polygons using Google Earth
#      - convert those polygon outlines into a raster image with positive values (1) inside the polygon and zero values outside of the polygon
#  - write the neural network that will be trained on the training data
#  - compare the performance of a full-connected neural network with a convolutional neural network in predicting if a given pixel will be part of an oil well.
# 
# Improvements:
#  - download the images direclty from a cloud service like AWS as discussed here - https://geohackweek.github.io/raster/04-workingwithrasters/
#  
#  
# As a reminder to myself, I am currently working on: 
#  - Creating template raster to burn the well polygons from Google earth to so I can create a raster version.
#  
# Mistakes
#  - Not ensuring the data has the same CRS is a HUGE mistake. Spent hours figuring that one out....
#%% [markdown]
# ## Importing and cropping the image file
#%% [markdown]
# This file has been downloaded from an imaging site. In the future, I will download it directly from AWS or Google Cloud.

#%%
#opening the raster file
im1 = '/Users/dustin/CS/data/Sat-images/Bakken/S2B_MSIL1C_20190606T174919_N0207_R141_T13UFP_20190606T194941.SAFE/GRANULE/L1C_T13UFP_A011750_20190606T174937/IMG_DATA/T13UFP_20190606T174919_B03.jp2'
satdat = rio.open(im1)


#%%
#show the image
show(satdat, cmap='gray')

#%% [markdown]
# You can't see much in the image above because the brightness of the cloud makes everything else dark. The axes values are in the projected units of the image. Let's look deeper into the attributes of the image.
#%% [markdown]
# ### Image Information

#%%
print('Image details in projected units')
print('Bounding box: ', satdat.bounds)
print("Width : {}".format(satdat.bounds.right - satdat.bounds.left))
print("Height: {}".format(satdat.bounds.top - satdat.bounds.bottom))
print('Image details in pixel units. Rows: {}, Columns: {}'.format(satdat.height, satdat.width))


#%%
print('Size of each pixel in projected units (meters in this case)')

#Calculating the size of each pixel in projected units
xres = (satdat.bounds.right - satdat.bounds.left) / satdat.width
yres = (satdat.bounds.top - satdat.bounds.bottom) / satdat.height
print("Column pixel size: {} m, Row pixel size: {} m"
      .format(xres, yres))
print("Are the pixels square: {}".format(xres == yres))


#%%
print('The CRS of the image is {}'.format(satdat.crs))
print('The metadata for the image is: {}: '.format(satdat.meta))


#%%
# Upper left pixel
up_left = (0,0)

# Lower right pixel.  Rows and columns are zero indexing.
down_right = (satdat.height - 1, satdat.width - 1)

print("Top left corner coordinates: {}".format(satdat.transform*up_left))
print("Bottom right corner coordinates: {}".format(satdat.transform*down_right))


#%%
#To explore the attributes of the rasterio object check the .__dir__() attribute table
#print(satdat.__dir__())


#%%
#converts to numpy array
img = satdat.read().squeeze()

print('Number of bands: ', satdat.count)
print('Number of indexes: ', satdat.indexes) #doesn't exist for single band
print('Number of dimentions: ', img.ndim)
print('Data type: ', img.dtype)
print('Object type: ', type(img))
print('Shape: ', img.shape)

##bad idea, image is too big to generate historgram
##plt.hist(img, bins = [0,20,40,60,80,100]) 
##plt.title("histogram") 
##plt.show()

#%% [markdown]
# ### Cropping the image file 
#%% [markdown]
# The image file is a bit too big for what I want to focus on, so I have cropping it based on the coordinates specified below.

#%%
#the cropping area of interest in WGS84 coordinates
minx, miny =   47.754309, -103.295
maxx, maxy =  47.865643, -103.030516 

bbox = box(minx, miny, maxx, maxy)

#creating a geopandas dataframe of the bounding box for the crop
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=4326)

#reprojecting into the raster image coordinate system - epsg:32613
geo_proj = geo.to_crs(epsg=32613)
geo_proj


#%%
#cropping the raster image
out_img, out_transform = mask(satdat, shapes=geo_proj.geometry, crop=True)
out_transform


#%%
out_img.dtype


#%%
#modify the cropped image's metadata
out_meta = satdat.meta.copy()
#parse the EPSG code
epsg_code = int(satdat.crs.data['init'][5:])
print(epsg_code)

out_meta.update({"driver": "JP2OpenJPEG", 
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform} )
out_meta


#%%
#write the cropped raster

out_jp2 = '/Users/dustin/CS/Projects/oil_well_detector/sat_images/T13UFP_B03_crop.jp2'
with rio.open(out_jp2, "w", **out_meta) as dest:
    dest.write(out_img)

#%% [markdown]
# ## Creating the training data
#%% [markdown]
# After creating the area of interest, we must construct the y-labels which will tell whether a given pixel is inside an oil well. The creation of these y-labels will be done by drawing polygons around oil wells in Google Earth and then converting these polygons into a raster image so that the pixels that are contained in an oil well will have an on label (1) and those pixels outside the oil well will have an off label (0).

#%%
well_polygons = gpd.read_file(r'/Users/dustin/CS/projects/oil_well_detector/non_images/training_data/Wells_polygon_93_20190808.gpkg')
well_polygons.crs


#%%
#we need to project the polygons that outline the wells into the CRS of the cropped image
well_proj = well_polygons.to_crs(epsg=32613)
well_proj.crs

#%% [markdown]
# Now the polygons need to be burned to a raster. I need to create a template raster to burn the polygons to. 

#%%
#Burning the polygon layers into a raster file

out_fn = '/Users/dustin/CS/Projects/oil_well_detector/sat_images/well_labels.jp2'

rst_fn = '/Users/dustin/CS/Projects/oil_well_detector/sat_images/T13UFP_B03_crop.tif'
rst = rio.open(rst_fn)
meta = rst.meta.copy()
meta.update(compress='lzw')

with rio.open(out_fn, 'w', **meta) as out:
    
    #we create a generator of geom, value pairs to use in the features.rasterize function
    shapes = ((feature['geometry'], 1) for feature in well_proj.iterfeatures())
    
    labels = features.rasterize(shapes=shapes, out_shape=out_img[0].shape, transform=meta['transform'], fill=0, dtype='uint16', all_touched=False)    
#    out.write_band(1, burned)
    out.write(burned, indexes=1)
show(labels)
print('Number of positive pixels:', np.sum(labels))
print('Number of total pixels: {}'.format(labels.shape[0]*labels.shape[1]))
print('Ratio of positive to total pixels: {}'.format(np.sum(labels)/(labels.shape[0]*labels.shape[1])))

#%% [markdown]
# As can be seen above, the number of positive pixels to the total is very small (0.6%), which means the data set is heavily imbalanced towards the negative case. Any learning algorithm will be tempted to just always predict a negative (zero) value since that represents 99.4% of all situations.
#%% [markdown]
# In the machine learning approaches I will use, the data will be fed into a fully connected neural network. So, the raster input and label images will be unrolled into vectors. 

#%%
#unroll the images into vectors
vec_img = np.reshape(out_img, -1)
print(vec_img.shape)
vec_labels = np.reshape(labels, -1)
print(vec_labels.shape)

#Creating the training data
test_ratio = 0.1 #ratio of the total dataset in the test set
dev_ratio = 0.1  #ratio of the total datset in the development set
train_ratio = 1 - test_ratio - dev_ratio

perm = np.random.permutation(vec_img.shape[0])

X_train = vec_img[perm][0:2000000]


#%%
np.random.permutation(5)

#%% [markdown]
# ## Creating the machine learning algorithm
#%% [markdown]
# Now, the fun part. The input values of the training data - X values - have been created by cropping the satellite image and the labels of the training data - Y values - have been created by converting the polygons outlining the wells into a raster image. We can now use the input X's and labeled Y's to train a machine learning algorithm to predict whether or not a given pixel is part of an oil well. 
#%% [markdown]
# ### Reshaping the data
#%% [markdown]
# 

#%%


#%% [markdown]
# Use F1 score for measuring accuracy

#%%



#%%


#%% [markdown]
# # Scratch
#%% [markdown]
# Notes:
# W103.388 to W103.028
# 
# N47.8548 to N47.755
# 
# 
# To do
#      - stitch raster’s together
#         • use buffer around points to cut out raster images of well sites to make y=1 values
#     -  make make cutouts of other imagees for y = 0 values - keeping them all the same size
#         • split into training and dev sites
#     - train algorithm with just one band
#     - take new images and cut them up with standard sizing and use a steping window to perform detection

#%%
def importImages(filenames):
    
    #imports the images

    satdat_list = []
    for  i in filenames:
        satdat = rio.open(i)
        satdat_list.append(satdat)
    
    return satdat_list


 
def changeCRS():
    #set or change the CRS of an image. Not sure if I need this
    
    pass


#%%
    def file_merge(files, )
    '''
    Description:
    Takes a list of image paths as input and merges all of the images
    into a single file
    
    Input:
    files -- a list of pathnames of the different images
    '''
        
    merge_files= [file1, file2]
    rio_files = []
    out_fp = '/Users/dustin/CS/Projects/Well_identification/Images/mosaic.tif'


    for i in merge_files:
        src = rio.open(i)
        rio_files.append(src)
        
    mosaic, out_trans = merge(rio_files)


#%%
def addBuffer(size, filename):
    '''
    Description: 
    Adds a buffer of size size
    
    Input:
    buffer size -- the size of the buffer in who knows what units
    filename  -- a string of the img pathname
    
    Output: 
    
    
    
    '''
    
    gdf = gpd.read_file(r'/Users/dustin/CS/Projects/Well_identification/well-sites.gpkg', layer='well-sites')

    buf = gdf.geometry.buffer(size)
    
    sq_buf = buf.envelope
    
    sq_buf.to_file("sq_buffer.shp")
    
    return  buff


#%%
#writing the mosaic'ed file 
with rio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)


#%%
#non-functional rasterize code - don't think it's Python
gdal_rasterize -burn 255 -l tessellate well_polygons.gpkg work.tif


