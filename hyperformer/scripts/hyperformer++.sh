# This scripts trains hyperformer++.

# We experimented with `reduction_factor` of 32, 16 and report the results of the model obtaining the 
# best results on the validation set on the test set.
python3 ./finetune_t5_trainer.py configs/hyperformer++.json 
