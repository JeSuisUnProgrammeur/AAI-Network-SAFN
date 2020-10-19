# auxiliary-feature-fusion-network-speech-inversion

The pytorch code implementation of the paper: 

SPEAKER-INDEPENDENT ACOUSTIC-TO-ARTICULATORY INVERSION BASED ON SPEECH SEPARATION AND AUXILIARY FEATURE FUSION

![The AFFN model we proposed is as:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-network-speech-inversion-/blob/main/figure1.jpg)

![The defferent experiment scenaire are as:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-fusion-network-speech-inversion-/blob/main/Experimentsetting.png)

The following details are details:

pytorchtools.py contains of some important method functions.

model_learning.py and train_learning.py are model and training in No-Fine-tuning scenaire.

![The Speech Separation Module pre-trained is as:](https://github.com/JeSuisUnProgrammeur/auxiliary-feature-network-speech-inversion-/blob/main/figure3.jpg)

model_learning_F01.py is the Speech Separation Module which is pre-trained.

train_learn_F01.py, model_learning.py and model_learning_F01.py are processing of model and training in Fine-tuning scenaire.
