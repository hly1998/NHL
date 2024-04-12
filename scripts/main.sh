python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet --mode simple --info CSQ # CSQ
python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ # CSQ+NHL w/o AD
python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --analysis # CSQ+NHL w/o D
python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1 # CSQ+NHL w/o A
python bit_length.py --device 0 --dataset cifar10 --test_map 4 --net ResNet_RML --mode RML_E --info CSQ --distill --distill_weight 1 --analysis # CSQ w/ NHL
 