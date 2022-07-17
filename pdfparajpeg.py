import os
import numpy
from os import path
from glob import glob

import numpy
from pdf2image import convert_from_path


def find_jpg(fdir):
    arfiles = []
    for dirpath, dirnames, filenames in os.walk(fdir,
                                                topdown=True):  # colocar o directorio das folhas de presença
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            arfiles.append(os.path.join(dirpath, filename))
    return arfiles

def find_pdf(fdir):
    arfiles = []
    for dirpath, dirnames, filenames in os.walk(fdir,
                                                topdown=True):  # colocar o directorio das folhas de presença
        for filename in [f for f in filenames if f.endswith(".pdf")]:
            arfiles.append(os.path.join(dirpath, filename))
    return arfiles


# count = 1

# arfiles = find_pdf("/home/nardo/Desktop/ProjetoFolhaPresencas/teste")
# print(len(arfiles))
# print(numpy.transpose(arfiles))

# for i in arfiles:
#     print("getting " + i)
#     pages = convert_from_path(i)
#     for page in pages:
#         page.save('p' + str(count) + '.jpg', 'JPEG')
#         print(count)
#         count += 1





