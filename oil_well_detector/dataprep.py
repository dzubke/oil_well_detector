import rasterio as rio
import geopandas as gpd
import shapely.geometry as shpgeo
from rasterio.mask import mask as riomask
from rasterio.plot import show as rioshow

def readimg(filename):
    '''Reads in a file with the filename 'filename' and returns a rasterio object

    Inputs:
    filename -- the filename of the image

    Outputs:
    satdat -- a rasterio object of the read file

    '''

    satdat = rio.open(filename)
    
    return satdat

def printinfo(satdat, showplot=False):
    '''Prints the information of the raster image 'satdat'

    Input:
    satdat -- a raterio object of a raster image
    showplot -- a boolean value that determines of the image will be plotted

    Output:
    Nothing is outputed. This function only prints the data

    '''
    print('Image details in projected units')
    print('Bounding box: ', satdat.bounds)
    print("Width : {}".format(satdat.bounds.right - satdat.bounds.left))
    print("Height: {}".format(satdat.bounds.top - satdat.bounds.bottom))
    print('\n')
    print('Image details in pixel units')
    print('Rows: {}, Columns: {}'.format(satdat.height, satdat.width))
    print('Number of bands: ', satdat.count)
    print('Number of indexes: ', satdat.indexes) #doesn't exist for single band
    
    if showplot :
        rioshow(satdat, cmap='gray')


def pixelres(satdat):
    '''Takes in a rasterio object and returns the sizel of a pixel in projected units

    Input:
    satdata -- a rasterio object

    Output:
    pixsize -- a list of integers of the [x-size, y-size] of the pixel in projected units
    punits -- a string that states what the projected units of the rasterio object
    boolsquare -- a boolean of whether the pixels are square (i.e. of equal length and width)

    '''

    xres = (satdat.bounds.right - satdat.bounds.left) / satdat.width
    yres = (satdat.bounds.top - satdat.bounds.bottom) / satdat.height

    pixsize = [xres, yres]
    boolsquare = xres == yres
    punits = satdat.crs.linear_units

    return pixsize, punits, boolsquare

def imgcrop(satdat, box, out_fn):
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
    
    out_img, out_transform = riomask(satdat, shapes=geo_proj.geometry, crop=True)

    #modify the cropped image's metadata
    out_meta = satdat.meta.copy()

    out_meta.update({"driver": "GTiff", 
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform} )
    
    with rio.open(out_fn, "w", **out_meta) as dest:
        dest.write(out_img)
    
    return out_img