# Speech-Decomposition-Auxiliary-Feature-Feature-Transformer-Network

The pytorch code implementation of the paper: 

Speaker-independent Acoustic-to-Articulatory Inversion based on Speech Decomposition and Auxiliary Feature

The SAFN model we proposed is as:

![The SAFN model we proposed is as:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-fusion-network-speech-inversion-/blob/main/figure1.jpg)


The defferent experiment scenaire are as:

![The defferent experiment scenaire are as:](https://github.com/JeSuisUnProgrammeur/AFFN/blob/main/Experiment%20setup.png)

The sesult of the experiment is:

![The sesult of the experiment is:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-fusion-network-speech-inversion-/blob/main/figure4.jpg)
The following details are details:

pytorchtools.py contains of some important method functions.

model_learning.py and train_learning.py are model and training in No-Fine-tuning scenaire.

The Speech Decomposition Network pre-trained is as:

![The Speech Decomposition Network pre-trained is as:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-fusion-network-speech-inversion-/blob/main/figure3.jpg)


model_learning_F01.py is the Speech Decomposition Network which is pre-trained.

train_learn_F01.py, model_learning.py and model_learning_F01.py are processing of model and training in Fine-tuning scenaire.
