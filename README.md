# ProjetoFolhaPresencas

### Projecto Licenciatura DEIS-ISEC
### José Lopes - 2019130541
***
## Table of Contents
1. [Prerequisites](#project-description)
2. [Project description](#users-manual)
3. [User's manual](#users-manual)
4. [Advice for use](#advice-for-use)



#### Previous work: [GitHub](https://github.com/renatogomes17/ProjetoFolhaPresencas)

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
| Tessaract|4.1.1| 
| Tensorflow & Keras |2.8.0| 
| Pillow |9.0.1| 

It's useful to have all the frameworks used available at your local machine if you want to test it:

`$ pip install -U scikit-learn numpy opencv-python pandas pyzbar seaborn pytesseract imutils pdf2image`

It is also required to have Tessaract installed: [Tesseract](https://tesseract-ocr.github.io/tessdoc/Home.html)
Tesseract has some dependencies so it is recommended to read the installation manual.


## Project description
* Input: One or more images of attendance sheets;
* Ouput: A .csv file with the students present in a given lesson.

The aim of this project is to develop a functional prototype for reading, through computer vision, of attendance sheets in classes:
* read student ID
* automate the process of recognizing signatures on attendance sheets 
* validate attendance. 
The application's primary goal is to identify the students that were present in each class and that have signed the attendance sheet.

This project will be combined with another to establish the connection of the application to the NONIO system. This means that the output of this program will be redirected to an API that will automatically register the students present in each class. That was or will be developed, in parallel, to route the data regarding the presence of the students in class to the NONIO system, so these features are not relevant in this project.

## User's manual
The program is executed from the command line: 
`$ python main.py [path]`

The argument with the directory path is optional. If left blank, the program starts from the project directory where the program runs. If a specific directory is entered, it is validated, and all subdirectories are scrolled through. If valid, it puts in an array all the file addresses that correspond to files of type .jpeg. Otherwise, the programme ends

Checks if the file "modelsignature.joblib" that corresponds to the model for classifying signatures exists in the directory or its subdirectories. If the file does not exist, the model is trained using the datasets - make sure they are in the root directory of the project in a folder called "input".


## Advice for use
For a correct reading, the sheets must comply with some requirements:
* Minimum resolution: 300 dpi. 400 dpi or more is recommended for best results.
* Orientation preserved
* Black and white
* It should not be hole-punched
