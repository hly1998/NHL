# NHL
The codes for a paper, we provides our codes and 10 deep hashing baselines.

# Main Dependencies
+ pytohn 3.8
+ torch 1.11.0+cu113
+ numpy 1.22.4
+ pandas 2.0.3

# How to run
You can easily run our code by following steps:

+ Replace "{your root}" in the file "utils/tools.py" with your own file path.
+ We have prepered some cases in scripts/main.sh, you counld run the command "sh scripts/main.sh" to begin the training process. 
+ Note that SHCIR and MDSH models should fisrt build their hash centers. We have prepred their hash centers in tmp_file/*, or you can using SHCIR_cg.py and SHCIR_cls.py to build the hash centers of SHCIR and bit_length/MDSH_cg.py to build the hash centers of MDSH.

# The explanations of main options
+ --device: choose the used cuda.
+ --dataset: select a dataset from [cifar10, imagenet, coco]
+ --info: choose a deep hashing model [CSQ, DBDH, DCH, DHN, DPN, DSH, DTSH, LCDSH, MDSH, SHCIR]
+ --mode: with or without NHL
+ --analysis: if True, use adaptive weight strategy
+ --distill: if True, use long-short cascade self-distillation
+ --distill_weight: the weight for long-short cascade self-distillation


