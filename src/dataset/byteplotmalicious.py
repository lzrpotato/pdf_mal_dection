'''
This program reads in a PDF file, converts it into an integer array (from the ASCII values of the bytes) and generates a grayscale image based on a given width
Author: Jack Zemlanicky
Contributors: Bill Luo
'''
import array as arr
from PIL import Image
import numpy as np
from numpy import ceil, sqrt
import cv2
import os


# functions

# main function to convert a pdf into a grayscale image using the byte plot strategy
def convert(directory,name,width):
    # Initialize our integer array
    int_array = arr.array('i')
    # Get the file as a byte stream
    pdfFile = open(directory+name,'rb')
    # Get the array of bytes from the stream
    byteArray = pdfFile.read()
    #print(byteArray)
    # Put the individual bytes (ascii values) into an array
    for byte in byteArray:
        int_array.append(byte)
    # If the given width is larger than or equal to the square root of the length of the int array, just pad zeroes then draw the image
    if width >= sqrt(int_array.__len__()):
        pad_zeroes(int_array,width)
        return draw_image(int_array,width,name,directory)
    # Else the given width is smaller than the sqrt of the length of the array, so we need to first create the image normally, then compress the newly created image to the desired width
    else:
        big_width = find_next_square(int_array.__len__())
        pad_zeroes(int_array,big_width)
        draw_image(int_array,big_width,name,directory)
        return compress_image(width,name,directory)
        

# Pad extra zeroes to get to the next square 
def pad_zeroes(int_array,width):
    while int_array.__len__()<width**2:
        int_array.append(0)

# Given the size of the file, find the next square number 
def find_next_square(size):
    # ceiling value so it is always >= the size of the file, never smaller
    return int(ceil(sqrt(size)))

# Given an int array and width, draw a grayscale image
def draw_image(int_array,width,name,directory):
    img = Image.new('L',(width,width))
    img.putdata(int_array)
    # Needed separate line since you cannot use str.replace in f strings
    trimmed_name = name + ".png"
    path_name = f"{directory}{trimmed_name}"
    img.save(path_name)
    return(path_name)

# Takes the larger image and scales it down to the given width
def compress_image(width,name,directory):
    trimmed_name = name + ".png"
    path_name = f"{directory}{trimmed_name}"
    img = cv2.imread(path_name,cv2.IMREAD_UNCHANGED)
    dimension = (width,width)
    img = cv2.resize(img,dimension)
    cv2.imwrite(path_name,img)
    return(path_name)

#convert('Sample\\','10esnonresestatsnap.pdf',256)