# ProjetoFolhaPresencas

### Projecto Licenciatura DEIS-ISEC
### José Lopes - 2019130541
***
## Table of Contents
1. [Prerequisites](#project-description)
2. [Project description](#users-manual)
3. [User's manual](#users-manual)
4. [Advice for use](#advice-for-use)



#### Previous work: [GitHub](https://github.com/nardobap/ProjetoFolhaPresencas)

## Prerequisites

| Library       | Version used     | 
| ------------- |:-------------:| 
| Imutils |	0.5.4 |
| Numpy	|1.20.3|
| OpenCV	|4.5.3|
| Pandas	|1.3.2|
| Pdf2image	|1.16.0|
| Pytesseract|	0.3.8|
| Python	|3.8.8|
| Pyzbar |0.1.8
| Scikit-learn|	0.24.2|
| Seaborn|	0.11.2|
| Tesseract|4.1.1| 
| Tensorflow & Keras |2.8.0| 
| Pillow |9.0.1| 

It's useful to have all the frameworks used available at your local machine if you want to test it:

`$ pip install -U scikit-learn numpy opencv-python pandas pyzbar seaborn pytesseract imutils pdf2image tensorflow pillow`

The repo already includes the tesseract executable. If you are using the Windows operating system all you have to do is unzip 
the 'Tesseract-OCR' folder in the project root directory.

Otherwise it is required to have Tessaract installed: [Tesseract](https://tesseract-ocr.github.io/tessdoc/Home.html)
Tesseract has some dependencies so it is recommended to read the installation manual.


## Project description
* Input: One or more images of attendance sheets;
* Ouput: 
    *  A .csv file with the students present in a given lesson.
    *  Warnings identifying students whose signatures may not be genuine

The aim of this project is to develop a functional prototype for reading, through computer vision, of attendance sheets in classes:
* read student ID
* automate the process of recognizing signatures on attendance sheets 
* validate attendance. 
* validate signature and detect possible forgeries (once at least 5 genuine signatures per student have been introduced)

The application's primary goal is to identify the students that were present in each class and that have signed the attendance sheet as well as verify the legitness
of each signature.

This project will be combined with another to establish the connection of the application to the NONIO system. This means that the output of this program will be redirected to an API that will automatically register the students present in each class. That was or will be developed, in parallel, to route the data regarding the presence of the students in class to the NONIO system, so these features are not relevant in this project.

## User's manual
The program is executed from the command line: 
`$ python main.py`

The program checks if the file "modelsignature.joblib" that corresponds to the model for classifying signatures exists in the directory or its subdirectories. If the file does not exist, the model is trained using the datasets - make sure they are in the root directory of the project in a folder called "input".

A menu will be prompted to the user with several commands such as adding a new class to the system or validating attendance sheets. In order to choose an option the user has to type the corresponding operation number prompted in the terminal.

* Type 1 do add a class to the system
* Type 2 in order to read the attendance sheets
* Type 3 to train a signature recognition model and detect possible forgeries
* Type 4 to visualize the classes list
* Type 5 to safely close the app

## Advice for use
For a correct reading, the sheets must comply with some requirements:
* Minimum resolution: 300 dpi. 400 dpi or more is recommended for best results.
* Orientation preserved
* Black and white
* It should not be hole-punched
* The absences should be marked using a horizontal line
* In order to assure high accuracy predictions the minimum recommended amount of genuine signatures is 10
