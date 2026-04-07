#!/bin/bash
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --trials 100

python3 ./test_gpt.py --alphabet "z y x w v u t s r q p o n m l k j i h g f e d c b a" --interval_size 1 --trials 100
python3 ./test_gpt.py --alphabet "z y x w v u t s r q p o n m l k j i h g f e d c b a" --interval_size 2 --trials 100

python3 ./test_gpt.py --alphabet "a b c d w x y z l m n o p e f g q r s h i j k t u v" --interval_size 1 --trials 100
python3 ./test_gpt.py --alphabet "a b c d w x y z l m n o p e f g q r s h i j k t u v" --interval_size 2 --trials 100

python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --trials 100

python3 ./test_gpt.py --alphabet "a b c d g i v u j r x h m q n y z t e w k p f o l s" --interval_size 1 --trials 100
python3 ./test_gpt.py --alphabet "a b c d g i v u j r x h m q n y z t e w k p f o l s" --interval_size 2 --trials 100