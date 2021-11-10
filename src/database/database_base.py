from typing import List


class TableBase():
    def __init__(self):
        pass

    def customized_entries(self):
        return NotImplemented

    def table_entries(self) -> List:
        self.table_name = 'results'
        self.table_entries = [
            ('exp','integer','not null'),
            ('nclass','integer','not null'),
            ('dnn','text','not null'),
            ('dataset','text','not null'),
            ('stride','real','not null'),
            # ***********************
            # ***********************
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