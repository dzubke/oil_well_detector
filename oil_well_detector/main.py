


if __name__ == '__main__':
    import dataprep
    import trainingdata

    im1 = '/Users/dustin/CS/data/Sat-images/Bakken/S2B_MSIL1C_20190606T174919_N0207_R141_T13UFP_20190606T194941.SAFE/GRANULE/L1C_T13UFP_A011750_20190606T174937/IMG_DATA/T13UFP_20190606T174919_B03.jp2'

    #importing the image as a rasterio object
    satdat = dataprep.readimg(im1)
    
    #getting some information on the raster image
    dataprep.printinfo(satdat)
    dataprep.pixelres(satdat)
    
    #creating the bounding box and cropping the image based on the box
    box = (47.754309, -103.295, 47.865643, -103.030516 )
    #the file name of the cropped image
    crop_fn = '/Users/dustin/CS/projects/oil_well_detector/data/sat_images/T13UFP_B03_crop_v2.tif'
    crop = dataprep.imgcrop(satdat, box, crop_fn)
    print('crop shape: {}'.format(crop[0].shape))
    
    crop1 = dataprep.readimg(crop_fn)

    dataprep.printinfo(crop1)

    poly_fn = '/Users/dustin/CS/projects/oil_well_detector/data/training_data/Wells_polygon_93_20190808.gpkg'
    
    polygon = trainingdata.polyread(poly_fn, satdat.crs.to_epsg())

    train_fn = '/Users/dustin/CS/projects/oil_well_detector/data/sat_images/well_labels_v2.tif'
    trainingdata.poly2raster(polygon, crop_fn, train_fn, crop[0].shape)

    #storing the metadata for the labels in lbl_meta and reshaping the labels as a vector in lbl_vec
    y_vec, y_meta = trainingdata.reshape(train_fn)
    x_vec, x_meta = trainingdata.reshape(crop_fn)

    print(y_vec.shape, x_vec.shape)
