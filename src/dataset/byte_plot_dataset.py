from .byteplot import convert
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import pickle
import logging 
from tqdm import tqdm
logger = logging.getLogger('dataset.byteplot')

path_name_benign = 'dataset/CLEAN_PDF_9000_files/'
path_name_malicious = 'dataset/MALWARE_PDF_PRE_04-2011_10982_files/'

class PDFDataset(Dataset):
    def __init__(self,plot_type='byte_plot', width=256):
        # call the __init__() of its parent class
        super().__init__()
        # We need to know if the incoming pdfs are converted to grayscale images using the byte plot or markov plot
        self.plot_type = plot_type
        self.width = width
        self._load()
        #for i, x in enumerate(self.X):
        #    if x is None:
        #        logger.info(f'index {i} {x}')
        # do something to initialize the pdf dataset object
        self.class_to_index = {'benign':0, 'malicious': 1}
        self.nc = 1
        self.nclass = 2
        self.key_name = 'img_byteplot'

    def __len__(self):
        # return the number of instances of the dataset
        # how to reference of this list like you did in your dataset?
        return len(self.labels)

    def __getitem__(self, idx):
        # return X, y, which is the array at index idx and the label (benign or malicious) at idx
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        X = self.X[idx]
        y = self.labels[idx]
        X, y = self._to_tensor(X, y)
        return X, y

    def _to_tensor(self, X, y):
        tensorX = torch.tensor(X, dtype=torch.float32) 
        tensorY = torch.tensor(y, dtype=torch.long)
        tensorX = tensorX.unsqueeze(0)
        #print(tensorX.shape)
        return tensorX,tensorY
    
    # Get the actual dataset pdfs and load them into a dictionary(good and bad pdfs)
    def _load(self):
        # In a sample directory with only 10 pdfs for now
        benign_files = (path_name_benign)
        malicious_files = (path_name_malicious)
        # Dictionary to store each plot for good and bad pdfs
        data_by_type = {'benign':None,'malicious':None}
        # check for pickled data file
        if os.path.isfile('temporary/byteplot_input.pickle'):
            logger.info("Pickled data detected. Skipping bulk i/o and unpickling data...")
            labelunpickler = open("temporary/byteplot_label.pickle",'rb')
            labels = pickle.load(labelunpickler)
            self.labels = labels
            inputunpickler = open("temporary/byteplot_input.pickle",'rb')
            X = pickle.load(inputunpickler)
            self.X = X
            logger.info("Done loading!")
            logger.info(f'X {len(self.X)} label {len(self.labels)}')
            return
    
        
        # only 2 key/value pairs in the dictionary, 'benign' and 'malicious'. Each value will be a list containing x and y path names to corresponding grayscale images
        if self.plot_type == 'byte_plot':
            # Convert all pdfs to images and save their paths in a list
            for label, path in {'benign': benign_files, 'malicious': malicious_files}.items():
                files = sorted(os.listdir(path))
                path_list = []
                count = 0
                for file_name in tqdm(files):
                    if file_name in ['log.txt','log.png']:
                        continue
                    if not file_name.endswith('png'):
                        if file_name.endswith('pdf'):
                            trimmed_name = file_name.replace("pdf","png")
                        else:
                            trimmed_name = file_name + '.png'
                        
                        if trimmed_name not in files:
                            #converts any pdf into a byteplot
                            path_name = convert(path,file_name,self.width)
                            logger.info(f'trimmed_name {trimmed_name} {path_name}')
                            path_list.append(cv2.imread(path_name,cv2.IMREAD_UNCHANGED))
                            
                        else:
                            path_name = f"{path}{trimmed_name}"
                            path_list.append(cv2.imread(path_name,cv2.IMREAD_UNCHANGED))
                        count += 1
                data_by_type[label] = path_list
                logger.info(f'[counter] {label} {count}')
        # Creates one large list by concatenating 2 smaller lists. Each smaller list has size = total number of values under the corresponding key in the data_by_type dictionary(benign or malicious). The large list (labels) has the same size as the entire dataset, and consists of x entries of "malicious" and y entries of "benign"
        labels = []
        file_type = {'benign':0,'malicious':1}
        for k in data_by_type.keys():
            n = len(data_by_type[k])
            label = np.repeat(file_type[k],n)
            labels.extend(label)
        self.labels = labels
        # Create the text file to store the data
        logger.info("Created file for pickling the labels")
        # pickle the data and dump it into the text file
        with open("temporary/byteplot_label.pickle",'wb') as fh:
            pickle.dump(labels, fh)

        # Implement list of inputs, X
        X = []
        for k in data_by_type.keys():
            X.extend(data_by_type[k])
        self.X = X
        logger.info("Created file for pickling the inputs")
        with open("temporary/byteplot_input.pickle",'wb') as fh:
            pickle.dump(X,fh)


