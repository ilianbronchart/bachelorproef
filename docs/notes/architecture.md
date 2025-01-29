# Model Pipeline Architecture Ideas

## The problem:

Ideally, we would define a catalog of target objects that we want to detect in a video, and have the software detect, segment, and track these automatically.


<!-- Note: could maybe improve accuracy of pipeline to detect frames where there is a large amount of motion blur, and skip these for grounding -->

## Options for the model pipeline:

### Option 1: Pretraining

1. We gather a dataset of videos of only the target objects. We segment and track the object throughout the video, and then train a model like YoloV5 on these segmented images.

#### Advantages:

1. Yolov5 is a very fast model, and could run in real-time on a GPU.
2. Less work at video analysis time

#### Disadvantages:

1. More work to set up the simulation with pretrained models
2. It could be that objects which have been moved relative to the pretraining data are not detected well.
3. We would need to retrain the model for each new object we want to detect.
4. We need to make sure to get a good dataset (multiple angles, lighting conditions?) to train on.

### Option 2: Zero-shot detection

1. We use a model like Grounding DINO to detect objects zero-shot via prompts
2. We use SAM 2 to track the detected objects

#### Advantages:

1. No need to pretrain on a dataset of the target objects
2. Could be more robust to changes in the environment since it's grounded in large foundation models
3. Could be more generalizable to new objects

#### Disadvantages:

1. Need to filter away false positives from the grounding model during analysis time (more work while analyzing the video)
2. Analysis could be slower than with a model like Yolov5

### Option 3: Manually click on target objects in video, then track

1. We manually click on the target objects in the video
2. We use a tracking model like SAM 2 to track the objects

#### Advantages:

1. No pretraining needed
2. Could be more accurate than the other methods

#### Disadvantages:

1. More work at video analysis time

### Option 4: Eyetracking as a first-class citizen + Grounding DINO

1. We use the eyetracking data to segment all objects the student looked at in the video
2. We crop out these objects and use a zero-shot detection model like Grounding DINO to label the objects
3. We filter away unwanted objects

How do we choose which points to sample for SAM?
Do we pick a region around the gaze point and sample from there?
How do we choose the size of the region?

#### Advantages:

1. We already start with a catalog of objects the student looked at
2. Could be more robust to changes in the environment since it's grounded in large foundation models
3. Could be more generalizable to new objects
4. Per-frame segmentation using SAM 2 is very fast

#### Disadvantages:

> Investigate accuracy of GazeSAM paper

## Grounding using a Vision API like GPT-4o

Our grounding images are small, but there are many of them, how much would this cost?
Could we use structured inputs and outputs to allow the model to choose from a list of possible answers?

### Option 5: Eyetracking as a first-class citizen + Content based image retrieval

See https://www.reddit.com/r/opencv/comments/b8f4up/question_scale_invariant_template_matching/

Preparation:
1. We use the glasses to record video of each target object we want to detect
2. We use SAM to track the objects in the video and put the objects in a Content Based Image Retrieval (CBIR) database

Analysis:
3. We use the eyetracking data to segment all objects the student looked at in the video using SAM
    - Based on the average error of the headset, sample points around the gaze point for mask detection. 
    - See https://connect.tobii.com/s/article/eye-tracker-accuracy-and-precision?language=en_US
4. We filter away unwanted objects (too large, too many components, ...)
5. We crop out the objects (how?) and use the CBIR database to label the objects
6. We now have a few keypoints for each object and use these to track the objects throughout the video 
7. We calculate when, for how long and how often the student looked at the objects using some metric

#### Advantages:

1. We already start with a catalog of objects the student looked at
2. More preparation time needed, but less work at video analysis time

#### Disadvantages:

1. The pipeline is more complex and errors could pile up

#### Feature Detectors:

SURF, FlannBasedMatcher