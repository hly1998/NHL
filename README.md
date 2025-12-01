# NHL
The codes for paper **Nested Hash Layer: A Plug-and-play Module for Multiple-length Hash Code Learning**, we provides our codes and some deep hashing model baselines.

# Main Dependencies
+ pytohn 3.8
+ torch 1.11.0+cu113
+ numpy 1.22.4
+ pandas 2.0.3

# How to run
You can easily run our code by following steps:

+ Replace "{your root}" in the file "utils/tools.py" with your own file path.
+ We have prepered some cases in scripts/main.sh, you counld run the command "sh scripts/main.sh" to begin the training process. 
+ Note that MDSH models should fisrt build hash centers. We have prepred their hash centers in tmp_file/*, or you can use bit_length/MDSH_cg.py to build the hash centers of MDSH.

# The explanations of main options
+ --device: choose the used cuda.
+ --dataset: select a dataset from [cifar10, imagenet, coco]
+ --info: choose a deep hashing model [CSQ, DBDH, DCH, DHN, DPN, DSH, DTSH, LCDSH, SHCIR, MDSH]
+ --mode: with or without NHL
+ --analysis: if True, use adaptive weight strategy
+ --distill: if True, use long-short cascade self-distillation
+ --distill_weight: the weight for long-short cascade self-distillation


