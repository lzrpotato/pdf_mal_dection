drop table if exists results;
PRAGMA foreign_keys = OFF;
CREATE TABLE results (
    exp integer not null,
    nclass integer NOT NULL,
    dnn text NOT NULL,
    fold INTEGER NOT NULL,
    acc real,
    stopepoch integer,
    bestepoch integer,
    PRIMARY KEY (exp,nclass,dnn,fold)
);

/* insert or update*/
/*
insert into results (
    method,fold,dataset,acc,c1,c2,c3,c4
    ) 
values ('BU_RvNN',1,'Twitter15',0.2,0.2,0.8,0.8,0.8) 
on conflict(method,fold,dataset) 
do update set acc=excluded.acc,c1=excluded.c1,c2=excluded.c2,c3=excluded.c3,c4=excluded.c4;
*/