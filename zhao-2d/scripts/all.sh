# predicted mask
python main.py --config_path settings/v1_zhao_ACKTR-h200-n64-M7-rA-P.yaml
python main.py --config_path settings/v2_zhao_ACKTR-h400-n64-M7-rA-P.yaml
python main.py --config_path settings/v3_zhao_ACKTR-h1600-n64-M7-rA-P.yaml

# ground truth mask
python main.py --config_path settings/v1_zhao_ACKTR-h200-n64-M7-rA-T.yaml
python main.py --config_path settings/v2_zhao_ACKTR-h400-n64-M7-rA-T.yaml
python main.py --config_path settings/v3_zhao_ACKTR-h1600-n64-M7-rA-T.yaml
