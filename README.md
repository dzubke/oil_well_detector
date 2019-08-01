# oil-well-detection
Detections oil wells in the Bakken oil field based on satellite imagery

For this project, I will be training an image recognition algorithm to recognize if 
a pixel is part of an oil well. I will be analyzing various types of satellite images to conduct this analysis.

I am still working on this project. So far I have:

 - downloaded the relevant satelittle images (the raw images are too big to be in a github repo)
 - utilized Rasterio and Shapely analyze the features of the image and crop an area of interest

Still to come I need to:
 - Create the training data
     - make outline examples of oil wells in polygons using Google Earth
     - convert those polygon outlines into a raster image with positive values (1) inside the polygon and zero values outside of the polygon
 - write the neural network that will be trained on the training data
 - compare the performance of a full-connected neural network with a convolutional neural network in predicting if a given pixel will be part of an oil well.
