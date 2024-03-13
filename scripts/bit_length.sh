# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DCH
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ

# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DTSH

# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DTSH

# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --distill
# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH --distill
# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --distill

# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --distill
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH --distill
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --distill

# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight

# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight
# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight
# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight

# python bit_length.py --device 1 --dataset imagenet-2 --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 1 --dataset imagenet-2 --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 1 --dataset imagenet-2 --test_map 10 --net ResNet --mode simple --info DTSH

# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 1.0 0.5 0.1
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 1.0 0.5 0.1
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 1.0 0.5 0.1

# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 0.7 0.5 0.3
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 0.7 0.5 0.3
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 0.7 0.5 0.3

# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 0.6 0.5 0.4
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 0.6 0.5 0.4
# python bit_length.py --device 0 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 0.6 0.5 0.4

# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info CSQ --step_update

# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info CSQ
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ

# 2.20 之前用的load checkpoints有问题，现在重新训练！
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DTSH

# 2.22

# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DTSH --space

# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH

# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --analysis
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH --analysis
# python bit_length.py --device 0 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --analysis

# 测试wandb
# python bit_length.py --device 0 --dataset cifar10 --test_map 3 --net ResNet --mode simple --info CSQ

# python bit_length.py --device 1 --dataset cifar10 --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --analysis

# 2024.3.6 依次测试baseline能否正确运行

# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info CSQ

# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info CSQ --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DCH --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DHD --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DHD --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DHN --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DPN --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DTSH --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info MDSH --wandb
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info MDSH --wandb

# 2024.3.11 测试加入distill的效果
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1

# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ

# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DBDH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DCH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DHD --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DHN --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DPN --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info DTSH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info MDSH --wandb --distill --distill_weight 1

# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info CSQ --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DBDH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DCH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DHD --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DHN --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DPN --wandb --distill --distill_weight 1

# 2024.3.12 跑到这里，下面两个模型还没跑
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info DTSH --wandb --distill --distill_weight 1
# python bit_length.py --device 0 --dataset imagenet --test_map 8 --net ResNet_RML --mode RML_E --info MDSH --wandb --distill --distill_weight 1