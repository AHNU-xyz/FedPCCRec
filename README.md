# FedPCCRec

Install dependencies:

```bash
pip install -r requirement.txt

Training Global Model
python PFRMP.py --dataset yelp_test --mode fednebmask --epoch 2000 --local_epoch 5

Local Fine-tuning
python end_local_train --dataset yelp_test  --local_epoch 100
