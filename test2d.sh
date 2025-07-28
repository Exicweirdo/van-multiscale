L=16
for beta in `seq 0.1 0.1 1.0`; do

python3 ./main2d.py -o out --ham fm  --lattice sqr --L $L --beta $beta --model parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 1 --batch_size 10000 \
--max_step 9000 --print_step 100 --save_step 3000 --visual_step 1000
python3 ./main2d.py -o out --ham fm  --lattice sqr --L $L --beta $beta --model half_parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 1 --batch_size 10000 \
--max_step 9000 --print_step 100 --save_step 3000 --visual_step 1000
python3 ./main2d.py -o out --ham fm  --lattice sqr --L $L --beta $beta --model made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 1 --batch_size 10000 \
--max_step 9000 --print_step 100 --save_step 3000 --visual_step 1000
done