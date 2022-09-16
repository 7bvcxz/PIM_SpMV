#!/bin/sh
cuda=0

"""
for model in "ds2_4r"
do
	for reg_size in 128
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
"""

#for model in "ds2_0" "ds2_0r" "ds2_1" "ds2_1r" "ds2_2" "ds2_2r" "ds2_3" "ds2_3r" "ds2_4" "ds2_4r" "gnmt_enc_0" "gnmt_enc_0r" "gnmt_enc_1" "gnmt_enc_2" "gnmt_enc_3" "gnmt_dec_0" "gnmt_dec_1" "gnmt_dec_2"
for model in "ds2_1r"
do
	for reg_size in 64 128
	do
		for sparsity in 0.9 0.8 0.7 0.6
		do
			for bound_ratio in 0.3 0.2 0.18 0.16 0.14 0.12 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.025 0.02 0.01
			do
				echo "model: ${model}"
				echo "spars: ${sparsity}"
				echo "reg_s: ${reg_size}"
				echo "bnd_r: ${bound_ratio}"
				echo "algorithm 15"

				python3 run.py --ver=no_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=15 --num_ch=16 --num_ba=16 --model=${model} --bound_ratio=${bound_ratio}
			done
		done
	done
done
