import os
import sys
from pathlib import Path

cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging

from src.summary.summary import Summary

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('[summary_all]')

if __name__ == '__main__':
    # db_names = ['exp_cnn_lstm_sw_1.db','exp_cnn.db','exp_pcnn_lstm_sw_1.db',
    #     'exp_cnn_attn_sw_1.db','exp_pcnn_attn_sw_1.db','exp_FCN_CA_sw_1.db',
    #     'exp_cnn_tran_sw_1.db','exp_fcn_tran_sw_1.db','exp_CNN_CA_sw_1.db',
    #     'exp_MSFCNCA_sw_1.db', 'exp_MSFCN1CATH_sw_1.db','exp_MSFCN1CATHCA_sw_1.db',
    #     'exp_MSFCNCATH_sw_1.db','exp_MSFCNCATHCA_sw_1.db']

    db_names = []
    for fn in os.listdir('src/database/'):
        if fn.endswith('.db'):
            if 'final_exp' not in fn:
                continue
            db_names.append(fn)
    db_names = sorted(db_names,reverse=True)
    for db_name in db_names:
        s = Summary(db_name)
        s.savedata()
        s.get_average_folds()
