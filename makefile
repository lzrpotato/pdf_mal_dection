all:
	@echo "###################################"
	@echo "# Start with following target:"
	@echo "# [cnn] for CNN_LSTM"
	@echo "# [cslurm] for cleaning slurm output"
	@echo "# [clog] for cleaning lightning logs"
	@echo "###################################"

cnn:
	sbatch slurm/slurm_cnnlstm.sh

cslrum:
	rm slurm/slurm_out/*

clog:
	rm -r lightning_logs/*
