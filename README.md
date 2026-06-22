# 1. Exploring Hyperparameters
[Solution](01-hyperparameters/Hyperparameters_README.md)
# 2. Gathering a Dataset
I took the required pictures and annoted them. After that I moved all the images including the ones from the tutors into the ``gesture_dataset_wiretux`` folder and structured it like the ``gesture_dataset_sample`` folder.

## Dataset and Confusion-Matrix generation
I created ``genDataset.py`` form the ``hyperparameters.ipynb`` to generate the required model with the ``gesture_dataset_sample``.
<br>
I then rewrote this script to ``genMatrix.py`` that generates the Confusion-Matrix and saves it as  ``conf-matrix.png``

# 3. Gesture-based Media Controls
The detection box is shown in the middle of the window. You can to do one of the following hand gestures until the counter reaches 8 to trigger an action:
- like: Volume louder
- dislike: Volume quieter
- stop: Track skip
- fist: Audio play/pause
With q you can stop it at any time.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/OcE5Fe4c)
