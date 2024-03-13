# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DTSH
# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 0 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH

# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DCH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DTSH

# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH

# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info CSQ
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DCH
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DTSH

# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info CSQ --distill
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DCH --distill
# python bit_length.py --device 3 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DTSH --distill

# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --distill
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DCH --distill
# python bit_length.py --device 3 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DTSH --distill

# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DCH --heuristic_weight
# python bit_length.py --device 1 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML --info DTSH --heuristic_weight

# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DCH --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info DTSH --heuristic_weight

# 2024.2.10 看一下稳定性 => 存在0.02左右的波动

# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 1 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML --info CSQ --heuristic_weight

# 2024.2.11

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DTSH

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info CSQ
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DCH
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DTSH

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info CSQ --distill
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DCH --distill
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DTSH --distill

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --distill
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DCH --distill
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --distill

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight

# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight
# python bit_length.py --device 0 --dataset imagenet-2 --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DCH --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info DTSH --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML --info CSQ --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01

# 2.20 之前用的load checkpoints有问题，现在重新训练！
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DTSH

# 2.21 查看 RML_E条件下，权重影响

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 2.0 0.5 0.05

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 3.0 0.5 0.03

# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DCH --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info DTSH --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01
# python bit_length.py --device 1 --dataset imagenet --test_map 5 --net ResNet_RML --mode RML_E --info CSQ --heuristic_weight --heuristic_weight_value 5.0 0.5 0.01

# 2024.2.22 看一下不加batch norm的结果如何
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DTSH
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet --mode simple --info CSQ

# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DTSH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info DCH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet --mode simple --info CSQ

# 2024.2.26 analysis

# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --analysis
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DCH --analysis
# python bit_length.py --device 2 --dataset imagenet --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --analysis

# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --analysis

# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH

# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info CSQ --norm --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH --norm --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --norm --analysis


# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info CSQ --norm batch --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DCH --norm batch --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 10 --net ResNet_RML --mode RML_E --info DTSH --norm batch --analysis

# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info CSQ --norm all --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info DCH --norm all --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info DTSH --norm all --analysis

# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info CSQ --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info DCH --analysis
# python bit_length.py --device 2 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info DTSH --analysis

# python bit_length.py --device 1 --dataset cifar10 --test_map 2 --net ResNet_RML --mode RML_E --info CSQ --analysis
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info CSQ
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info DCH
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info DHD
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info DHN
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info DPN
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info DTSH
# python bit_length.py --device 1 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML --info MDSH

# 2024.3.12 测试analysis效果
# python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ
python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --analysis --info CSQ --distill_weight 1

