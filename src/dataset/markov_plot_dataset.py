"""
Tyler Nichols
New Mexico Tech 2021 REU - Lab 2, pt. 2
Dataset & Dataloader

Current Restrictions:
    1. requires input of data folder's name
    2. data must be in the same directory as the py file
    3. input files must only include one type of file each
    4. must input number of each type of file in input folder(s)

Helpful Sources:
    https://realpython.com/working-with-files-in-python/
    https://github.com/lzrpotato/pdf_mal_dection/tree/main/src/dataset
"""
from .markovplot import PDF_markovPlot

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import logging
logger = logging.getLogger('dataset.markovplot')

""" ======================= Access Input File ======================= """    
def loadData(pdfPaths, pdfLabels):
    load = True
    while load:
        # take in data's info (folder, type, and count)
        folderName = input("Enter the name of your desired data's folder: ")
        dataType = input("Enter data type (benign/b or malicious/m): ")
        dataCount = int(input("Enter number of files of input type: "))
        # prepare to load in data
        basePath = os.getcwd() + "/" + folderName + "/"
        i = len(pdfPaths)
        # load in paths
        with os.scandir(folderName) as entries:
            for entry in entries:
                pdfPaths[i] = (basePath + entry.name)
                i += 1
        # load in labels
        if (dataType == "benign") or (dataType == "b"):
            for i in range(dataCount):
                pdfLabels.append(0.0)
        elif (dataType == "malicious") or (dataType == "m"):
            for i in range(dataCount):
                pdfLabels.append(1.0)
        # check if loading more data
        loadMore = input("load more data? (yes/y or no/n): ")
        if (loadMore == "no") or (loadMore == "n"):
            load = False
""" ---------------------- NO INPUT VERSION ----------------------
 takes 3 LIST inputs: folder names, file types, and file counts """ 
def loadDataNoIn(pdfPaths, pdfLabels, folders, types):
    for i in range(len(folders)):
        # take in data's info (folder, type, and count)
        folderName = folders[i]
        dataType = types[i]
        # prepare to load in data
        basePath = folderName + "/"
        count = 0
        # load in paths
        for entry in os.listdir(folderName):
            if entry in ['log.txt','log.png']:
                continue
            if entry.endswith('.png'):
                continue
            pdfPaths.append(basePath + entry)
            # load in labels
            if (dataType == "benign") or (dataType == "b"):
                pdfLabels.append(0)
            elif (dataType == "malicious") or (dataType == "m"):
                pdfLabels.append(1)
            count += 1
        logger.info(f'[count] {count}')
    
""" =========================== PDF Dataset =========================== """
# ----------------------- INPUT VERSION -----------------------
class PDFDataset(Dataset):
    def __init__(self, nbyte = 1):
        # call the __init__() of its parent class
        super().__init__()
        # create a list of the file's inputs
        self.name = 'pdf_Dataset'
        self.nbyte = nbyte
        self.pdfPaths = []
        self.pdfLabels = []
        loadData(self.pdfPaths, self.pdfLabels)

    def __len__(self):
        # return the number of instances of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # return markov plot & its label
        markov = PDF_markovPlot(self.pdfPaths[idx])
        """ %%%%%%%%%%% NOT SURE WHAT TO DO WITH THE LABEL YET %%%%%%%%%%% """
        label = self.pdfLabels[idx]
        markov, label = self._to_tensor(markov, label)
        return markov, label

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
""" ========================== FUNCTION VERSION ========================== 
    folders = list of input folders
    types = list of types of files in input folders
    counts = list of  number of files in each input folder                 """
class PDFDatasetNoIn(Dataset):
    def __init__(self, nbyte = 1):
        # call the __init__() of its parent class
        super().__init__()
        # create a list of the file's inputs
        
        self.name = 'pdf_Dataset'
        self.nbyte = nbyte
        self.pdfPaths = []
        self.labels = []
        
        self._load_markov_plot()
        self.nc = 1
        self.key_name = 'img_markovplot'
        self.nclass = 2
        self.class_to_index = {'benign': 0, 'malicious': 1}
    
    def _load_markov_plot(self):
        
        if os.path.isfile('temporary/markovplot_input.pickle'):
            self.X = pickle.load(open('temporary/markovplot_input.pickle','rb'))
            self.labels = pickle.load(open('temporary/markovplot_label.pickle','rb'))
            return

        folders = ['dataset/CLEAN_PDF_9000_files','dataset/MALWARE_PDF_PRE_04-2011_10982_files']
        types = ['b','m']
        loadDataNoIn(self.pdfPaths, self.labels, folders, types)
        
        X = []
        for p in tqdm(self.pdfPaths):
            X.append(PDF_markovPlot(p))
        self.X = X
        with open("temporary/markovplot_input.pickle",'wb') as fh:
            pickle.dump(self.X, fh)

        with open("temporary/markovplot_label.pickle",'wb') as fh:
            pickle.dump(self.labels, fh)

    def __len__(self):
        # return the number of instances of the dataset
        return len(self.pdfPaths)

    def __getitem__(self, idx):
        # return markov plot & its label
        #markov = PDF_markovPlot(self.pdfPaths[idx])
        """ %%%%%%%%%%% NOT SURE WHAT TO DO WITH THE LABEL YET %%%%%%%%%%% """
        label = self.labels[idx]
        x = self.X[idx]
        markov, label = self._to_tensor(x, label)
        markov = torch.unsqueeze(markov, dim=0)
        return markov, label

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
