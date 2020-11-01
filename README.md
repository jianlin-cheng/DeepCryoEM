# DeepCryoPicker - a deep learning tool for picking protein particles in cryo-EM images
Deep learning methods for CryoEM data analysis
Cryo-electron microscopy (Cryo-EM) is widely used in the determination of the three-dimensional (3D) structures of macromolecules. Particle picking from 2D micrographs remains a challenging early step in the Cryo-EM pipeline due to the diversity of particle shapes and the extremely low signal-to-noise ratio (SNR) of micrographs. Because of these issues, significant human intervention is often required to generate a high-quality set of particles for input to the downstream structure determination steps. 

# Datasets
cryo-EM Micrographs that been used in this repostory have been collected from:
  - The first dataset is "EMPIAR-10146"- Apoferritin tutorial dataset for cisTEM, Dataset description is avaliable in https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10146/#&gid=1&pid=1
  
  - The second dataset has both Top and Side-view called (KLH), the KLH Dataset is available Online, http://nramm.nysbc.org/.
  - The third datatset has shape that is considered as an irregularly shaped protein, EMPIAR-10028-80S ribosome, the dataset is downloaded from https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10028/
  - The fourth dataset has complex protein particle shapes (EMPIAR-10017-Beta-galactosidase), the dataset is downloaded from https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/

In general, this repostogy has three main folders as follow:
- The first folder is the datasets in which the four different datasets have been collected.
- Second folder is the "Component 1: Fully Automated Training Particle Picking-Selection based Unsupervised Learning Approach"
  - This folder has the two our presious models which can do the following steps:
    - Stage 1: Fully Automated Single Particle-Picking.
    - Stage 2: Fully Automated Training Particles-Selection. 
      - Perfect “good” Top and Side-view Training Particle Selection using AutoCryoPicker: Unsupervised Learning Approach for Fully Automated Single Particle Picking in Cryo-EM Images, which is used mainly for top and side view training particles picking and selection. 
      - Perfect “good” Irregular and Complex Training Particle-Selection using SuperCryoEMPicker: A Super Clustering Approach for Fully Automated Single Particle Picking in Cryo-EM, which is used for irregular and complicated training particles picking  and selection. 

- The third folder is the "Component 2: Fully Automated Single Particle Picking based on Deep Classification Network", which has two models;
  - First mone is the Deep Classification Neural Network (Training Model).
  - Second one is the Automated Single Particle Picking (Testing Model). 

# Requirements
-You need to have a MATLAB 2017 (a)/(b) or the latest MATLAB version. 

# How to Run
- To run this repostory you need to follow the following steps:
  - The first matlab code folder is the "Pre-processing Stage" which is used to preprocessed the whole images dataset and plot the average results of the PSNR, SNR, and MSE, ans well as to the student-t test. 
  - The second matlab code is the "Signle Particle Detection_Demo" which is the single particle picking without the GUI version. 
  - To run this task you have to go to the main matlab file "AutoPicker_Demo1" just you need to update the dataset folder directoty and CLICK run in matlab. 
  - In this case the program will as you to select one single image then the program will auotomatically runs and display the single particles detection and picking. 
  - Finally, there is a GUI version called "Guide User Interface_GUI" which is all in one, you need just to go directly to the "AutoCryoPicking" or "AutoCryoPicking" then run it. 
  - the system will asks again to upload one single cryo-EM image then there is some other options such as: 
  - Load cryo-EM : for load any7 cryo-EM for testing.
  - Pre-processing (cryo-EM) : for doing the preprocessing task for the tested image. 
  - Particles Detection and Picking: for detect and picking the particles in the tested image. 
  - Performance Results: In this case - if you want to get the accuracy results and aother measurement you have to have a GT for each tested image we have already provide two images. 
  - in this case, we have to select the GT image and the system will automatically calculate and display all the performnace results once you click of the "Particles Picking Accuracy" - cryo-EM projection: This task is to extract the BOX for each single particle. 
  - Export Particles: This task is to extract the box dimension and the particle center information to *.TXT file.

# Main Manuscript
- The main manuscript that describe the DeepCryoPicker is avaliable at: https://www.biorxiv.org/content/10.1101/763839v1.
- Please cite this work as: "DeepCryoPicker: Fully Automated Deep Neural Network for Single Protein Particle Picking in cryo-EM
Adil Al-Azzawi, Anes Ouadou, Highsmith Max R, John J. Tanner, Jianlin Cheng doi: https://doi.org/10.1101/763839".
