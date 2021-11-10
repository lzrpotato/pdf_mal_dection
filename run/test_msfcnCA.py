import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging
import time
import traceback


jobid = os.environ.get('SLURM_JOB_ID')


logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

if not os.path.isdir('logging'):
    os.mkdir('logging')

if jobid is not None:
    filehandler = logging.FileHandler(f"logging/msfcnCA_{jobid}_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
else:
    filehandler = logging.FileHandler(f"logging/msfcnCA_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

def test():
    from src.trainer.msfcnCA_trainer import train, setup_arg
    args = setup_arg()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    try:
        train(args)
    except Exception:
        logger.error(f"Error: {traceback.format_exc()}")


if __name__ == '__main__':
    test()
    