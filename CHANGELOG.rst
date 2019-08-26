=========
Changelog
=========

While I figure that the changelog should be used to log changes to the code, or mainly to articulate updates to different versions, I am going to use it 
as a way of keeping track of what I am working on. 


20190825
===========
 - The tensorflow neural network model doesn't seem to be working well given that there are so few positive pixels
 - This may be due to an inability of the model to work with the data it's being given. The data is heavily imbalanced with only 0.6% of the pixels being 
    positive. Therefore, a better algorithm may be one that identifies outliers instead of a binary classification.
 -  It is also possible that analyzing the data as 0/1 based on the pixel is inferior to say marking entire pictures of the wells as being 0/1 and using
    a sliding window across the raster images. 
 - The sliding window approach seems complicated, so I would prefer to stick with just identifying pixels. 
 - However, it would be wise to do some literature search to see what approaches and algorithms have been used. 
 - One final thought would be to include the pixel location as an input as the 1's are all clustered together. 

 - More or less, I have built a first approach and it failed. So, what should I do? I think a literature search would be a good idea to see what is out there
    rather than just continuing to try different approaches. 
 - I think I won't bother to get the F1-statistic working because all the predictions will be zero's which will give it a high F1-stat because 99.4% of the 
    data are zeros.
