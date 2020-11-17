Project Title: Myoelectric Control of Robotics via Machine Learning Interpretation


Getting Started:
1) combinedFeatureExtraction.py  *Used in research 19/20*
The first file that you will use is combined feature extraction. This file
is used to extract specific features from the raw MESs taken from the DAQ.

To use this file, place the file path of the raw data into the variable PATH.
Place the file path of where you want the extracted features to populate in PATH2
and pick a name for the new file in the following line.


2) seperateFeatureExtraction.py *Not yet used*
This file is similar to combinedFeatureExtraction.py except that if places each
feature into a separate file instead of a combined file.

To use this file, place the file path of the raw data into the variable PATH.
Place the file path of where you want the extracted features to populate in PATH2.


3) Algorithms5_13.py
This file runs several algorithms on the data created in
combinedFeatureExtraction.py.

To use this file, place the file of the extracted features into the variable
PATH2.


Authors: Preston Robinette, Payton Phelps, Eli Owens
