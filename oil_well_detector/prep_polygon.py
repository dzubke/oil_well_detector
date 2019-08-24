import geopandas as gpd
import rasterio as rio
from rasterio import features
from typing import List

def polygon_read(in_fn: str, dest_crs: int) -> gpd.GeoDataFrame:
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

def polygon_2_raster(polygon: gpd.GeoDataFrame, rst_fn: str, out_fn: str, out_shape:List[int]) -> None:
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
            
            labels = features.rasterize(shapes=shapes, out_shape=out_shape, transform=meta['transform'], fill=0, dtype='uint16', all_touched=False)    
            out.write(labels, indexes=1)
