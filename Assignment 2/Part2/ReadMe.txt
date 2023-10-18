Install all the required modules mentioned in requirements.txt

Python version: 3.8.15


Part 2

Before running the python file, place shape_predictor_68_face_landmarks.dat file for automatic face landmark detection and marking.

1. To run : python part2_faceswap.py



Additional Steps:

On prompt, python file asks for the path to images (Images should be named I1.jpg, I2.jpg)

Initially, we need to select 4 points similar to Part 1 where pose correction and image resizing occur. Then, we can go with dlib for automatic point selection or choose manual point selection. 

Once images are swapped, you are asked to select a rectangular area of the top left corner and right bottom corner for a swatch from which 50 colours are selected, which helps in colour transfer. 

Once swatches are selected on both images, we swap the faces.