# Myolectric Control of Prosthetics and Robotics via Machine Learning Interpretation
This library is used for research on the myoelectric control of prosthetics and robotics via machine learning interpretation. In an effort to increase the functionality of prosthetic and robotics, this research focuses on increasing dexterity by identifying/classifying individual finger movements, a challenge due to the interconnected nature of the muscles of the forearm and the noninvasive nature of electrodes.

A myoelectric signal (MES) is a voltage difference measured across two muscles as shown in the figure below. The two electrodes across the muscles of the forearm are measuring the MES and the third electrode across the boney part of the elbow is used to ground the signal.

![Electrode Diagram](/Images/Electrode_Diagram.png)

The signals from the electrodes on the arm are then read by a PCB which takes the differential voltage, or MES, filters the signal, and amplifies the signal to be read by an Arduino. A diagram of the circuit is on the left which shows the components of the circuit and their functions. The physical PCB is pictured on the right.

![Circuit Diagram](/Images/circuit_diagram.png)
![Circuit](/Images/circuit.png)

The amplified MES from the PCB is then read into an Arduino which controls the 3D printed hand by triggering attached servos. Once activated, the servos rotate, pulling their attached strings which are run the 3D printed hand's fingers like tendons, and closing the hand into a fist. Once activated again, the servos rotate back, releasing the tension in their strings and allowing the hand to open again. The figure below shows the 3D printed hand and the servos attached on the back.

![3D Printed Hand](/Images/3D_hand.png)

Previous research has accomplished the response of the hand to opening and closing, and the results can be seen [here.](https://www.youtube.com/watch?v=ljKoZNYS_Rw) This research works to improve upon this past model by implementing control for individual fingers.

## [Circuit](https://github.com/pkrobinette/hand/tree/main/Circuit%2019:20)
- KiCAD files for the PCB with surface mounted components used in this research.

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
- Houses the files that extract `Data`. The extracted features from the data can either be combined into one file or sorted into seperate files based on their labels. The extracted features are then implemented and tested on various ml supervised learning models.

## [Results](https://github.com/pkrobinette/hand/tree/main/Results)
- Contains Confusion Matrix images for the various ml models implemented and tested. 

![Decision Tree Confusion Matrix](/Results/DT_confusion.png)

**Acknowledgements** Presbyterian College Department of Physics, NASA SC Space Grant Consortium, SC-INBRE, SCICU
