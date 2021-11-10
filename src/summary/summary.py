import logging

from numpy.lib.function_base import average
from src.database.expe_dm import Database
import pandas as pd
import os

logger = logging.getLogger('src.summary.summary')



class Summary():
    def __init__(self, db_name):
        self.db = Database(db_name=db_name)
        self.db_name = db_name
        self.results = None

    def read_database(self):
        self.results = self.db.get_all_as_dataframe()
        logger.info(f'print database {self.db_name} \n {self.results}')
        return self.results
    
    def get_average_folds(self):
        results = self.db.get_average_folds()
        logger.info(f'average results over nfold\n{results}')

    def savedata(self):
        if not self.results:
            self.read_database()
            if not os.path.isdir('results/'):
                os.mkdir('results/')
            if self.results is not None:
                self.results.to_csv(f'results/database={self.db_name}.csv')

    
class summary_dataset():
    def __init__(self):
        self.datapath = {'benign':'./dataset/CLEAN_PDF_9000_files/','mal':'dataset/MALWARE_PDF_PRE_04-2011_10982_files/'}

    def statistic(self):
        fsizes, fsize_by_category, labels_count = self.load_metadata_info()
        df_fsize = pd.DataFrame(fsizes,columns=['fsize'])
        logger.info(f'\n[Fsize stat]\nmax {df_fsize.max().values[0]} bytes \nmean {df_fsize.mean().values[0]} bytes \nmin {df_fsize.min().values[0]} bytes')
        logger.info(f'\n[File label count]\n{labels_count}')
        logger.info(f"\n[Fsize stat by benign]\nmax {max(fsize_by_category['benign'])} bytes \nmean {average(fsize_by_category['benign'])} bytes \nmin {min(fsize_by_category['benign'])} bytes")
        logger.info(f"\n[Fsize stat by mal]\nmax {max(fsize_by_category['mal'])} bytes \nmean {average(fsize_by_category['mal'])} bytes \nmin {min(fsize_by_category['mal'])} bytes")

    def load_metadata_info(self):
        fsizes = []
        fsize_by_category = {}
        labels_count = {}
        for t, d in self.datapath.items():
            labels_count[t] = 0
            fsize_by_category[t] = []
            for fn in os.listdir(d):
                if fn.endswith(('.png','.log')):
                    continue
                filesize = os.path.getsize(os.path.join(d,fn))
                fsizes.append(filesize)
                fsize_by_category[t].append(filesize)
                labels_count[t] += 1

        return fsizes, fsize_by_category, labels_count
