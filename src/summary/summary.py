import logging
from src.database.expe_dm import Database

logger = logging.getLogger('src.summary.summary')



class Summary():
    def __init__(self, db_name):
        self.db = Database(db_name=db_name)
        self.db_name = db_name

    def read_database(self):
        df = self.db.get_all_as_dataframe()
        logger.info(f'print database {self.db_name} \n {df}')
    