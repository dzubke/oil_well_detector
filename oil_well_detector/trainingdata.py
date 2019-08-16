import geopandas as gpd
import rasterio as rio
from rasterio import features as riofeatures
import numpy as np


def polyread(in_fn, dest_crs):
    '''This function read a filename 'in_fn' of a geopackage layer that contains polygons
        surrounding the oil wells.

    Input:
    in_fn -- the filename of the geopackage containing the polygons outlining the oil wells
    dest_crs -- an integer which specifies the coordinate reference system of the destination 
            raster image

    Output:
    proj_poly -- a geopandas object of the polygons projected in the 'dest_crs' crs

    '''

    polygons = gpd.read_file(in_fn)

    proj_poly = polygons.to_crs(epsg=dest_crs)

    return proj_poly

def poly2raster(polygon, rst_fn, out_fn, out_shape):
    '''This function take in a geopandas object of polygons and converts them into a raster
        image based on the 'temp_rast' raster

    Input:
    polygon -- the geopandas object of polygons
    rst_fn -- the filename of the raster that will be the template upon which the polygons will be burned
    out_fn -- the output file name that the raster of the polygons will be written to
    out_shape -- a two element list of the x and y shape of the output raster image

    Output:

    '''

    with rio.open(rst_fn, 'r') as rst:
        meta = rst.meta.copy()
        meta.update(compress='lzw')

        with rio.open(out_fn, 'w', **meta) as out:

            #we create a generator of geom, value pairs to use in the features.rasterize function
            shapes = ((feature['geometry'], 1) for feature in polygon.iterfeatures())
            
            labels = riofeatures.rasterize(shapes=shapes, out_shape=out_shape, transform=meta['transform'], fill=0, dtype='uint16', all_touched=False)    
            out.write(labels, indexes=1)


def reshape(rst_fn):
    '''Takes in a raster image and reshapes it into a column vector

    Input:
    rst_fn -- a rasterio object whose values will be reshaped into a column vector
    
    Output:
    rst_vec -- a column vector of the the numpy array associated with the 'img' rasterio object
    metadata -- the metadata associated with the 'img' rasterio object

    '''
    with rio.open(rst_fn, 'r') as rst:
        metadata = rst.meta.copy()
        array = rst.read(1)
        rst_vec = array.reshape(-1,1)
        

    return rst_vec, metadata
