#!/bin/sh

for model in "ds2_" "ds2_" "ds2_"
do
	for reg_size in 64
	do
		for sparsity in 0.9 0.8 0.7 0.6
		do
			echo "model: ${model}"
			echo "spars: ${sparsity}"
			echo "reg_s: ${reg_size}"
		
			python3 run.py --ver=test_split --sparsity=${sparsity} --register_size=${reg_size} --part_col=1 --algorithm=12 --num_ch=16 --num_ba=16 --model=${model}
		done
	done
done
