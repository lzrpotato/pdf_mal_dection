import sqlite3 as lite
import os
import pandas as pd
from dataclasses import dataclass
import logging
logger = logging.getLogger("database.expe_dm")

@dataclass
class Results():
    exp: int
    nclass: int
    dnn: str
    dataset: str
    stride: float
    fold: int
    acc: float
    f1micro: float
    f1macro: float
    fbenign: float
    fmal: float
    stopepoch: int
    bestepoch: int
    label: str
    time: str


class Database():
    def __init__(self, db_name):
        self.db_name = db_name
        self.path = './src/database/'
        self.db_path = os.path.join(self.path,self.db_name)
        self.timeout = 10000

        self.register_type()
        self.setup_database()

    def register_type(self):
        lite.register_adapter(bool,int)
        lite.register_converter('bool',lambda v: int(v) != 0)

    def setup_database(self):
        #self.create_database()
        self.table_name = 'results'
        self.table_entries = [
            ('exp','integer','not null'),
            ('nclass','integer','not null'),
            ('dnn','integer','not null'),
            ('dataset','text','not null'),
            ('stride','real','not null'),
            ('fold','integer','not null'),
            ('acc','real','not null'),
            ('f1micro','real','not null'),
            ('f1macro','real','not null'),
            ('fbenign','real','not null'),
            ('fmal','real','not null'),
            ('stopepoch','integer',''),
            ('bestepoch','integer',''),
            ('label','text',''),
            ('time','text ','')
        ]
        self.columns = [e[0] for e in self.table_entries]
        #self.columns = ['exp','nclass','dnn','fold','dataset','acc','f1','stopepoch','bestepoch','label']
        self.keys = ['exp','nclass','dnn','dataset','stride','fold']
        create_tb_sql = self.build_create_table_sql(self.table_name,self.table_entries,self.keys)
        self.create_database(create_tb_sql)

    def build_create_table_sql(self, table_name,table_entries,keys):
        tb_entry = []
        # add columns to the table
        for c in table_entries:
            tb_entry.append(' '.join(filter(None,c)))
        # add primary keys to the table
        tb_entry.append('primary key (' + ', '.join(keys) + ')')
        # concatenate tb entry
        tb_entry_str = ',\n'.join(tb_entry)
        # generate create table 
        create_tb_sql = """PRAGMA foreign_keys = OFF;\n""" + \
                        """CREATE TABLE {} (\n""".format(table_name) + \
                        tb_entry_str + \
                        """\n);"""

        return create_tb_sql

    def create_database(self,create_tb_sql):
        if not os.path.isfile(self.db_path):
            #os.system(f'sqlite3 {os.path.join(self.path,db_name)} ";"')
            self.create_table(create_tb_sql)

    def connect(self):
        conn = lite.connect(self.db_path,detect_types=lite.PARSE_DECLTYPES)
        return conn

    def create_table(self, create_table_sql):
        logger.info('[database] create table')
        
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.executescript(create_table_sql)
            conn.commit()
        except lite.Error as e:
            logger.error('[create_table] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()

    def save_results(self, results: Results):
        logger.info('[database] save results')
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % int(self.timeout))
            cur.execute('begin')

            insert_str = f"""insert into {self.table_name} ({','.join(self.columns)}) \n""" + \
                         f"""values ({','.join(['?' for i in range(len(self.columns))])}) \n""" + \
                         f"""on conflict({','.join(self.keys)}) \n""" + \
                         f"""do update \n""" +  \
                         f"""   set acc=excluded.acc \n""" + \
                         f"""       where acc < excluded.acc;"""
            
            cur.execute(insert_str, 
                        ([results.__dict__[c] for c in self.columns])
                    )
            conn.commit()
            logger.info(f'[save_results] successful\n {results}')
        except lite.Error as e:
            logger.error('[save_results] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
    def save_status(self, p):
        logger.info('[database] save_status')
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % int(self.timeout))
            cur.execute('begin')

            insert_str = f"""insert into {self.table_name} ({','.join(self.columns)}) \n""" + \
                         f"""values ({','.join(['?' for i in range(len(self.columns))])}) \n""" + \
                         f"""on conflict({','.join(self.keys)}) \n""" + \
                         f"""do update \n""" +  \
                         f"""   set acc=excluded.acc \n""" + \
                         f"""       where acc < excluded.acc;"""
            cur.execute(insert_str, 
                        ([p[c] for c in self.columns])
                    )

            conn.commit()
        except lite.Error as e:
            logger.error('[save_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
    
    def read_status(self, p):
        res = None
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name} where {' and '.join([k+'=?' for k in self.keys])};"""
            cur.execute(select_str,
                ([p[c] for c in self.keys])
            )
            
            res = cur.fetchone()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def check_finished(self, p):
        res = self.read_status(p)
        if res == None:
            return False
        return True
        
    def read_all(self):
        res = None
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name}\n""" + \
                         f"""order by {','.join(self.keys)};"""
            cur.execute(select_str)
            
            res = cur.fetchall()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def read_by_keys(self, query):
        res = None
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name}\n""" + \
                         f"""where { ' and '.join([k+'=?' for k in query.keys()]) }\n""" + \
                         f"""order by {','.join(self.keys)};"""

            cur.execute(select_str,
                tuple(query.values())
            )
            
            res = cur.fetchall()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def get_by_query_as_dataframe(self, query):
        res = self.read_by_keys(query)
        if res is None:
            return None

        df = pd.DataFrame(res, columns=self.columns)
        return df

    def get_all_as_dataframe(self):
        res = self.read_all()
        if res is None:
            return None
        
        df = pd.DataFrame(res, columns=self.columns)
        return df

    def delete_all_entry(self):
        conn = None
        is_success = True
        answer = input(f'Do you want to delete all results in {self.db_path} for table {self.table_name}? yes/no\n')
        if answer == 'yes':
            pass
        else:
            logger.info('exit without deleting')
            return
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""delete from {self.table_name};"""
            cur.execute(select_str)
            
            conn.commit()
            logger.info(f'records are deleted successfully.')
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))
            is_success = False
        finally:
            if conn:
                conn.close()
        
        return is_success


def _test():
    db_name = 'exp_test1.db'
    ss = Database(db_name)
    p = {'exp':1,'acc':1.0,'f1':1.0,'dnn':'CNN','nclass':4,'fold':0,'stopepoch':0,'bestepoch':0,'label':''}
    r = Results(**p)
    ss.save_results(r)
    ss.save_status(p)
    p = {'exp':2,'acc':1.0,'f1':1.0,'dnn':'CNN','nclass':4,'fold':0,'stopepoch':0,'bestepoch':0,'label':''}
    r = Results(**p)
    ss.save_results(r)
    ss.save_status(p)
    param = {'exp':1,'nclass':4,'dnn':'CNN','fold':0}
    res = ss.read_status(param)
    print(res)
    res = ss.check_finished(param)
    print(res)
    param = {'exp':1,'nclass':4,'dnn':'CNN','fold':1}
    res = ss.check_finished(param)
    print(res)
    res = ss.read_all()
    print(res)
    ss.get_all_as_dataframe()
    res = ss.read_by_keys({'exp':1,'dnn':'CNN','nclass':4})
    print(res)
    res = ss.get_by_query_as_dataframe({'exp':0,'dnn':'CNN','nclass':4})
    print(res)


if __name__ == '__main__':
    _test()

