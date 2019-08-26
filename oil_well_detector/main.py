


if __name__ == '__main__':
    import prep_raster, prep_polygon, prep_data, explore_raster, explore_data, nn_layers, nn_forward_prop, nn_compute_cost, nn_model
    import tensorflow as tf
    import sys
    import matplotlib.pyplot as plt


    im1 = '/Users/dustin/CS/data/sat_images/bakken/S2B_MSIL1C_20190606T174919_N0207_R141_T13UFP_20190606T194941.SAFE/GRANULE/L1C_T13UFP_A011750_20190606T174937/IMG_DATA/T13UFP_20190606T174919_B03.jp2'

    #importing the image as a rasterio object
    satdat = prep_raster.img_read(im1)
    
    #getting some information on the raster image
    explore_raster.raster_info(satdat)
    explore_raster.raster_pixelres(satdat)
    
    #creating the bounding box and cropping the image based on the box
    box = (47.754309, -103.295, 47.865643, -103.030516 )
    #the file name of the cropped image
    crop_fn = '/Users/dustin/CS/projects/oil_well_detector/data/sat_images/T13UFP_B03_crop_v2.tif'
    crop = prep_raster.img_crop(satdat, box, crop_fn)
    print(f'crop shape: {crop[0].shape}')
    
    crop1 = prep_raster.img_read(crop_fn)

    explore_raster.raster_info(crop1)

    poly_fn = '/Users/dustin/CS/projects/oil_well_detector/data/training_data/Wells_polygon_93_20190808.gpkg'
    
    polygon = prep_polygon.polygon_read(poly_fn, satdat.crs.to_epsg())

    train_fn = '/Users/dustin/CS/projects/oil_well_detector/data/sat_images/well_labels_v2.tif'
    prep_polygon.polygon_2_raster(polygon, crop_fn, train_fn, crop[0].shape)

    #storing the metadata for the labels in lbl_meta and reshaping the labels as a vector in lbl_vec
    Y_data, Y_meta = prep_data.reshape(train_fn)
    X_data, X_meta = prep_data.reshape(crop_fn)

    print(f"X_data shape: {X_data.shape}")
    print(f"Y_data shape: {Y_data.shape}")

 #   x_data_norm = prep_data.normalize(x_data) #this has an error

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = prep_data.data_split( X_data, Y_data, (0.8, 0.1, 0.1) )
   

    explore_data.describe(Y_dev)
    #explore_data.histogram(Y_dev.T)

 #   sys.exit()
    print("\nNow starting with the neural network layers")

    
    layers_dims = [X_train.shape[0], 200, 90, 10, 1]
    tf.reset_default_graph()
    with tf.Session() as sess:
        print("\nCreate placeholders for the input and label data")
        X, Y = nn_layers.create_placeholders(X_data.shape[0], Y_data.shape[0])
        print (f"X = {X}")
        print (f"Y = {Y}")

        print("\nInitialize the parameters")
        parameters = nn_layers.initialize(layers_dims, initialization='randn')
        for l in range(1,len(layers_dims)):
            print(f"W = {parameters['W'+str(l)]}")
            print(f"b = {parameters['b'+str(l)]}")

        print("\nCreate the computational graph")
        Z_L, cache = nn_forward_prop.forward_prop(X, parameters)
        print(f"Z_L: {Z_L}")
        
        print("\nCalculates the cost function")
        cost = nn_compute_cost.compute_cost(Z_L, Y)
        print(f"Cost: {cost}")

    print("\nRunning the model")
    parameters, Z_train, Z_test = nn_model.model(layers_dims, X_train, Y_train, X_test, Y_test, num_epochs=6)
    print("here I am")
    print("Z_train")
    explore_data.histogram(Z_train)
    explore_data.describe(Z_train)

    print("Z_test")
    explore_data.histogram(Z_test)
    explore_data.describe(Z_test)

    # as I somewhat expected, the nn is always predicting a zero value because there are so few positive 
    # values. I need to think about how to proceed....