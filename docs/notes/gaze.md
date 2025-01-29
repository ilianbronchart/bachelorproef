This file contains a list of notes taken about gaze tracking and gaze estimation.

Relevant keywords:
- saccades
- fixations
- smooth pursuits
- Vestibular ocular reflex  (VOR)


# Relevant Links:
- https://connect.tobii.com/s/article/Gaze-Filter-functions-and-effects?language=en_US
- https://connect.tobii.com/s/article/types-of-eye-movements?language=en_US

# Possible gaze filtering python libraries:
- https://github.com/DiGyt/cateyes

# Investigate?

Once we have real world gaze data, we can investigate the common parameters inside a simulation environment to tune the gaze filtering algorithm.
- How fast are the saccades?
- How fast are VOR movements?

# Converting gaze data from normalized x,y to angles: #INCLUDE
https://www.rpalmafr.com/post/calculating-visual-angles-in-eye-movement-data

# INCLUDE

It might be important to add settings for fov_x and fov_y incase the trainers upgrade their eye tracking hardware.
Since the gaze data is normalized (0-1), we can convert it to angles using the following formula:
angle = 2 * atan( (normalized_gaze - 0.5) * tan(fov/2) ) * 180/pi

# links

https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf
https://link.springer.com/article/10.3758/s13428-024-02360-0
https://connect.tobii.com/s/article/Gaze-Filter-functions-and-effects?language=en_US#GazeFilterI-VTAttention
https://pmc.ncbi.nlm.nih.gov/articles/PMC9699548/



!!!! Contains interesting insights
https://link.springer.com/article/10.3758/s13428-022-01833-4

```
In addition, studies have proposed systems incorporating conventional object recognition algorithms (ORAs) for automatic data annotation
(De Beugher et al., 2012; Toyama et al., 2012). For frame-by-frame images captured by a scene camera, these systems automatically extract
the region around the gaze location and compare its contents to the images of the preregistered object shapes using conventional feature-matching
algorithms such as scale-invariant feature transform (SIFT) (Lowe, 1999). This approach can automatically inform when relevant objects are gazed
upon without requiring designated markers or prescribed object positions. However, while effective on an ad hoc basis when exploring gaze behaviors
towards specific preregistered objects such as museum exhibits (Toyama et al., 2012), these systems lack flexibility as conventional ORAs are
essentially shape-matching algorithms that work only for preregistered objects with fixed shapes. Since the recognizable objects must exactly
match their pre-specified shapes, these systems cannot handle object types with inter-object (e.g., cars and buildings) or intra-object
variances (e.g., humans and animals). Notably, efforts have been made to include variable shapes such as human faces and bodies
(De Beugher et al., 2014) but the flexibility of such a system is not easily generalizable since it requires a set of designated ORAs,
each tailored for a specific object type.
```

### Gaze data quality

#INCLUDE
It seems like wearing glasses under the tobii eyetracking glasses yields very poor tracking results. (Confirm with Jorrit)
Ensuring a good calibration result is crucial for good SAM performance  from gazedata.


#INCLUDE Analysis of gaze depth data in tobii
https://thesis.unipd.it/retrieve/484d5830-8dff-45d1-8fbc-016807a5093f/Tesi_Magistrale__PD_%20%281%29.pdf

