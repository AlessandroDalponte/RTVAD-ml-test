## RTVAD-ml-test: Real-Time Video Anomaly Detection (machine learning test)

The purpose of this project is to merge, adapt and improve both KM3D detection ([**Banconxuan/RTM3D**](https://github.com/Banconxuan/RTM3D)) and multi-object tracking ([**adipandas/multi-object-tracker**](https://github.com/adipandas/multi-object-tracker)) codes, to create a system for tracking the spacial positions of objects detected from video captures of a single monocular RGB video camera. This project finds itself in its initial phase of conception and design, and still needs improvement.

## Description of test

- A video file is openned in frames, as arrays, which are then processed by the KM3D detector;
- 2D and 3D bounding boxes are added to the detected objects in each frame;
- The 2D bounding boxes are used as inputs by the tracker;
- An output video file is generated, containing the bounding boxes and tracks;
- The frames are also saved as .png image files for evaluation;

OBS: The tracker is still not implemented for Bird's Eye View.

The pre-trained weights "model_res18_2.pth" were chosen for this test. They presented the best results among the ones available in [**Banconxuan/RTM3D**](https://github.com/Banconxuan/RTM3D) for the ResNet architecture.

Two trackers were chosen for this test: "Centroid KF tracker" and "SORT".

A simple video file of cars on a road was used.

## Computer stats and versions

- Processor AMD Ryzen 5 3400g
- Graphics card NVIDIA GeForce GTX 1050
- Ubuntu 18.04
- GCC version 5.5.0
- Nvidia driver version 515.65.01
- CUDA version 9.0

## Installation and environment setup

Download this repository.

Install Anaconda in machine.

Create environment, suggested name 'RTVAD':
    
    conda create --name RTVAD python=3.6

Activate environment:
    
    conda activate RTVAD

Install Pytorch:
    
    conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch

Go to directory 'RTVAD-ml-test' and install requirements and setup:
    
    pip install -r requirements.txt

    pip install -e .

Go to directory 'RTVAD-ml-test/src/lib/models/networks/DCNv2' and compile deformable convolutional network:
    
    bash make.sh

Go to directory 'RTVAD-ml-test/src/lib/utils/iou3d' and compile iou3d:
    
    python setup.py install

## Before running code

The video file is read from the path 'RTVAD-ml-test/data/video/video.mp4'. In case you want to change the video file, go to the mentioned directory and name the new video as 'video.mp4'. You can also change the code in 'RTVAD-ml-test/src/faster.py' to read the desired file as you wish.

In the 'RTVAD-ml-test/data' directory, there are other subdirectories. In 'weights', the 'model_res18_2.pth' pre-trained weights file is located; in 'calib', the 'calib.txt' video camera calibration file is located; and in 'final_frames', 'final_video' and 'results_for_bev', the output video, output frames and output files for generating a Bird's Eye view (without tracks), can be found.

To choose the type of tracker, just assign the tracker name to the 'tracker' variable in 'RTVAD-ml-test/src/faster.py'. 'SORT' is set by default.

## Running the code

Go to 'RTVAD' folder (make sure the correct environment is active):
     
     python ./src/faster.py --calib_dir ./data/calib/ --load_model ./data/weights/model_res18_2.pth --gpus 0 --arch res_18

## Final considerations

- The weights used in this test were not trained for the specific case in which they are in use;
- The video camera calibration file for this test needs adjustments;
- The 'SORT' uses IOU as part of its tracking system; so in this case, because of the two considerations mentioned above, the 2D bounding box of the main vehicle is not stable between the initial frames, and it causes the tracker to change the vehicle's ID. In this case, the Centroid KF tracker works best, but might not be the best option for a model with specific trained weights and a better calibration camera file.
