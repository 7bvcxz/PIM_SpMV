#!/bin/sh
cuda=0

for model in "ds2_0" "ds2_" "ds2_"
do
	for reg_size in 64
	do
		for sparsity in 0.9 0.8 0.7 0.6
		do
			echo "model: ${model}"
			echo "spars: ${sparsity}"
			echo "reg_s: ${reg_size}"
			echo "algorithm 13"

			python3 run.py --ver=no_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=13 --num_ch=16 --num_ba=16 --model=${model}

			echo "model: ${model}"
			echo "spars: ${sparsity}"
			echo "reg_s: ${reg_size}"
			echo "algorithm 14"

			python3 run.py --ver=no_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=14 --num_ch=16 --num_ba=16 --model=${model}

			echo "model: ${model}"
			echo "spars: ${sparsity}"
			echo "reg_s: ${reg_size}"
			echo "algorithm 12"

			CUDA_VISIBLE_DEVICES=${cuda} python3 run.py --ver=test_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=12 --num_ch=16 --num_ba=16 --model=${model}
			echo "model: ${model}"
			echo "spars: ${sparsity}"
			echo "reg_s: ${reg_size}"
			echo "algorithm 17"

			CUDA_VISIBLE_DEVICES=${cuda} python3 run.py --ver=test_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=17 --num_ch=16 --num_ba=16 --model=${model}
		done
	done
done

