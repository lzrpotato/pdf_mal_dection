import torch
from torch.utils.data import Dataset, DataLoader


class PDFDataset(Dataset):
    def __init__(self):
        # call the __init__() of its parent class
        super().__init__()
        
        # do something to initialize the pdf dataset object
        pass

    def __len__(self):
        # return the number of instances of the dataset
        lenght = 100
        return lenght

    def __getitem__(self, idx):
        # return X, y
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        X, y= 1, 2
        X, y = self._to_tensor(X, y)
        return X, y

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def testcase_test_pdfdataset():
    dataset = PDFDataset()

    # setup dataloader
    # check pytorch document for the parameter list
    dl = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        )
    
    # loop through dataset
    for X, y in dl:
        # print out X and y 
        print(X, y)

if __name__ == '__main__':
    testcase_test_pdfdataset()