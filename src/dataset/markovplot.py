"""
Tyler Nichols
New Mexico Tech - REU 2021

This file will focus on the creating functions for facilitating the
conversion of a PDF to a Markov Plot
General Idea: PDF -> byte stream -> transition matrix -> markov plot
"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger('dataset.markovplot')


def byte_stream_to_int_array(bytes_array, nbyte):
    n = len(bytes_array)
    rem = n % nbyte
    npad = nbyte - rem
    # padding bytearray with zero
    bytes_array += bytes([0]*npad)
    integer_array = np.frombuffer(bytes_array, dtype=np.dtype(f'uint{nbyte*8}'))
    return integer_array

""" ============= PDF -> Byte stream -> Transition matrix ============= """
# Sort each byte's prevalence into a container
def PDF_transMatrix(pdfName):
    # construct nested list container
    transMatrix = np.zeros((256,256))
    # open bytestream
    with open(pdfName, "rb") as f:
        bytes_array = f.read()
    
    integer_array = byte_stream_to_int_array(bytes_array, nbyte=1)
    
    np.add.at(transMatrix, (integer_array[:-1], integer_array[1:]), 1)
    return transMatrix

def PDF_probMatrix(pdfName):
    transMatrix = PDF_transMatrix(pdfName)
    # initialize probability matrix
    sum = transMatrix.sum(axis=0)
    probs = np.divide(transMatrix, sum, out=np.zeros_like(transMatrix), where=sum!=0)
    return probs

def PDF_markovPlot(pdfName, createImage = False, imageName = ''):
    probMatrix = PDF_probMatrix(pdfName)
    # find max and min probability values
    probMax = np.max(probMatrix)
    probMin = np.min(probMatrix)
    # create pixel values & update probMatrix
    probMatrix = (probMatrix -probMin) / (probMax-probMin)
    
    # convert: pixelMatrix -> array -> image file
    if (createImage):
        pixelMatrix = np.array(probMatrix, dtype=np.uint8)
        # generate image file's name from input name
        pdfName = pdfName[0:len(pdfName) - 4]  # cut of .pdf file suffix
        pngName = imageName
        # generate name based on input file if no new name is input
        if imageName == '':
            char = ''
            for i in range(len(pdfName)):
                char = pdfName[len(pdfName) -1-i]
                if char == '\\' and pdfName[len(pdfName) -2-i] == '\\':
                    break
                pngName += char
            pngName = pngName[::-1] + '.png'
        new_image = Image.fromarray(pixelMatrix)
        new_image.save(pngName)
    return probMatrix

""" ################ PDF NAME INPUT & FUNCTION CALL ################ """
def main():
    #pdfName = input("Input name of PDF, including .pdf suffix: ")
    pdfName = 'testPDF.pdf'
    matrix = PDF_markovPlot(pdfName)
    print(matrix[0])
    
if __name__ == '__main__':
    main()
