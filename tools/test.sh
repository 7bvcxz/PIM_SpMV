python3 matrix_generator.py &&
cp model.pt ../test.pt &&
cd .. &&
python3 run.py --ver=no_split --sparsity=0 --part_col=1 --algorithm=16 --num_ch=2 --num_ba=2 --model=test --w_decay=1 --lr_init=1 &&
cd tools
