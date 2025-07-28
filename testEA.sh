for glass_seed in 12 42 72 102 132; do
python3 ./seqtempEA.py -o out_EA/seed_$glass_seed/ --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta_start 0.5 --beta_end 8 --beta_step 0.5 --model made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --batch_size 10000 \
--max_step 6000 --print_step 100 --save_step 6000 --visual_step 1000
python3 ./seqtempEA.py -o out_EA/seed_$glass_seed/ --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta_start 0.5 --beta_end 8 --beta_step 0.5 --model half_parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --batch_size 10000 \
--max_step 6000 --print_step 100 --save_step 6000 --visual_step 1000
python3 ./seqtempEA.py -o out_EA/seed_$glass_seed/ --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta_start 0.5 --beta_end 8 --beta_step 0.5 --model parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --batch_size 10000 \
--max_step 6000 --print_step 100 --save_step 6000 --visual_step 1000
done