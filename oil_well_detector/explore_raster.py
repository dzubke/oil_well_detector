from rasterio.plot import show


def raster_info(satdat, showplot=False):
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
        show(satdat, cmap='gray')


def raster_pixelres(satdat):
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