# 2024.3.5 用于跑baseline
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info CSQ --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DBDH --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DCH --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DHD --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DHN --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DPN --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info DTSH --wandb
# python bit_length.py --device 3 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info MDSH --wandb

python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info CSQ --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DBDH --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DCH --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DHD --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DHN --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DPN --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info DTSH --wandb
python bit_length.py --device 3 --dataset imagenet --test_map 8 --net ResNet --mode simple --info MDSH --wandb
