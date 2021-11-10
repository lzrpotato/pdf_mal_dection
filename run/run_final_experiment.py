import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging
import time
import traceback


def getlogger(args):
    jobid = os.environ.get('SLURM_JOB_ID')
    logger = logging.getLogger()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

    #handler = logging.StreamHandler(sys.stdout)
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)

    name = 'finexp'

    if jobid is not None:
        logpath = Path(f'logging/{name}/{args.model}/{jobid}/')
        logpath.mkdir(parents=True, exist_ok=True)
        
        filehandler = logging.FileHandler(f"logging/{name}/{args.model}/{jobid}/{time.strftime('%Y-%m-%d-%H:%M:%S')}_{args.fold}_{os.getpid()}.log")
    else:
        logpath = Path(f'logging/{name}/{args.model}/')
        logpath.mkdir(parents=True, exist_ok=True)
        filehandler = logging.FileHandler(f"logging/{name}/{args.model}/{time.strftime('%Y-%m-%d-%H:%M:%S')}_{args.fold}_{os.getpid()}.log")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger

def test():
    from src.trainer.final_experiment_trainer import train, setup_arg
    args = setup_arg()
    logger = getlogger(args)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    try:
        train(args)
    except Exception:
        logger.error(f"Error pid {os.getpid()}: {traceback.format_exc()}")
        print(f"Error pid {os.getpid()}: {traceback.format_exc()}")


if __name__ == '__main__':
    test()
    