# Myolectric Control of Prosthetics and Robotics
![Electrode Diagram](/Images/Electrode_Diagram.png)

## [Circuit](https://github.com/pkrobinette/hand/tree/main/Circuit%2019:20)
- KiCAD files for the PCB with surface mounted components

## [Data](https://github.com/pkrobinette/hand/tree/main/Data)
- Houses the data used in the analysis of individual finger movements. Signals are recorded for 7 different gestures:
  - **null** (no movement)
  - **fist** open and close
  - **thumb** open and close
  - **index** open and close
  - **middle** open and close
  - **ring** open and close
  - **pinkie** open and close

## [Scripts](https://github.com/pkrobinette/hand/tree/main/Scripts)
- 'Scripts' houses the files that extract 'Data'. The extracted features from the data can either be combined into one file or sorted into seperate files based on their labels. The extracted features are then implemented and tested on various ml supervised learning models.

## [Results](https://github.com/pkrobinette/hand/tree/main/Results)
- Contains Confusion Matrix images for the various ml models implemented and tested. 

![Decision Tree Confusion Matrix](/Results/DT_confusion.png)
