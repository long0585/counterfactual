#!/bin/bash
# interval 1 forward 
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'high'
# interval 1 random
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'high'
# interval 2 forward 
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high'
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'high'
# interval 2 random
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'low'
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'medium'
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high'
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 2 --gpt_engine 'gpt-5.4' --effort 'high'
# interval 1 forward composite 
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low' --composite
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'low' --composite
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium' --composite
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'medium' --composite
python3 ./test_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high' --composite
python3 ./analyze_gpt.py --alphabet "a b c d e f g h i j k l m n o p q r s t u v w x y z" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'high' --composite
# interval 1 random composite
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'low' --composite
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'low' --composite
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'medium' --composite
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'medium' --composite
python3 ./test_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --trials 100 --gpt_engine 'gpt-5.4' --effort 'high' --composite
python3 ./analyze_gpt.py --alphabet "x y l k w b f z t n j r q a h v g m u o p d i c s e" --interval_size 1 --gpt_engine 'gpt-5.4' --effort 'high' --composite