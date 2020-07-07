# DeepCryoEM
Deep learning methods for CryoEM data analysis
Cryo-electron microscopy (Cryo-EM) is widely used in the determination of the three-dimensional (3D) structures of macromolecules. Particle picking from 2D micrographs remains a challenging early step in the Cryo-EM pipeline due to the diversity of particle shapes and the extremely low signal-to-noise ratio (SNR) of micrographs. Because of these issues, significant human intervention is often required to generate a high-quality set of particles for input to the downstream structure determination steps. 

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
  
- The main manuscript that describe the DeepCryoPicker is avaliable at: https://www.biorxiv.org/content/10.1101/763839v1.
- Please cite this work as: "DeepCryoPicker: Fully Automated Deep Neural Network for Single Protein Particle Picking in cryo-EM
Adil Al-Azzawi, Anes Ouadou, Highsmith Max R, John J. Tanner, Jianlin Cheng doi: https://doi.org/10.1101/763839".
