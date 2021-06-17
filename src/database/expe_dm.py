import sqlite3 as lite
import os
from dataclasses import dataclass
import logging
logger = logging.getLogger("database.expe_dm")

@dataclass
class Results():
    exp: int
    nclass: int
    dnn: str
    fold: int
    acc: float
    f1: float
    stopepoch: int
    bestepoch: int


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
            ('fold','integer','not null'),
            ('acc','integer','not null'),
            ('f1','integer','not null'),
            ('stopepoch','integer',''),
            ('bestepoch','integer',''),
        ]
        self.columns = ['exp','nclass','dnn','fold','acc','f1','stopepoch','bestepoch']
        self.col_type = ['integer', 'integer','text','integer','real','real','integer','integer']
        self.col_null = ['not null','not null',' not null', 'not null','','','']
        self.keys = ['exp','nclass','dnn','fold']
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

            insert_str = f"""insert into results ({','.join(self.columns)}) \n""" + \
                         f"""values ({','.join(['?' for i in range(len(self.columns))])}) \n""" + \
                         f"""on conflict({','.join(self.keys)}) \n""" + \
                         f"""do update \n""" +  \
                         f"""   set acc=excluded.acc \n""" + \
                         f"""       where acc < excluded.acc;"""
            
            cur.execute(insert_str, 
                        ([results.__dict__[c] for c in self.columns])
                    )
            conn.commit()
        except lite.Error as e:
            logger.error('[save_status] error {}'.format(e.args[0]))

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

            insert_str = f"""insert into results ({','.join(self.columns)}) \n""" + \
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
            
            select_str = f"""select * from results where {' and '.join([k+'=?' for k in self.keys])};"""
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
        

if __name__ == '__main__':
    db_name = 'exp_test1.db'
    ss = Database(db_name)
    p = {'exp':1,'acc':1.0,'f1':1.0,'dnn':'CNN','nclass':4,'fold':0,'stopepoch':0,'bestepoch':0,}
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