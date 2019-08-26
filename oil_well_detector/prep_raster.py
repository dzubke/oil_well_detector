import rasterio as rio
import geopandas as gpd
import shapely.geometry as shpgeo
from rasterio.mask import mask

def img_read(filename: str):
    '''Reads in a file with the filename 'filename' and returns a rasterio object

    Inputs:
    filename -- the filename of the image

    Outputs:
    satdat -- a rasterio object of the read file

    '''

    satdat = rio.open(filename)
    
    return satdat


def img_crop(satdat, box, out_fn):
    '''Crops a raster image 'satdat' based on the boundaries in 'box' and writes it to the filename 'out_fn' and returns
        the raster image.

    Input:
    satdat -- rasterio object of a satellite image
    box  -- list of coordinates in standard GPS coordinates (epsg: 4326) that specify the bounding box 
        with format (minx, miny, maxx, maxy) where minx is the minimum x-value and maxy is the maximum y-value
    out_fun -- the filename for the output cropped image

    Output:
    satdat_crop  -- the satdat image croppped based on the boundaries specified in bbox

    '''

    bbox = shpgeo.box(*box)

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=4326)

    geo_proj = geo.to_crs(epsg=satdat.crs.to_epsg())
    
    out_img, out_transform = mask(satdat, shapes=geo_proj.geometry, crop=True)

    #modify the cropped image's metadata
    out_meta = satdat.meta.copy()

    out_meta.update({"driver": "GTiff", 
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform} )
    
    with rio.open(out_fn, "w", **out_meta) as dest:
        dest.write(out_img)
    
    return out_img