
for glass_seed in 12 42 72 102 132; do
    for beta in `seq 0.5 1 7.5`; do
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_parallel_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model half_parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_half_parallel_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    done

    for beta in `seq 1 1 8`; do
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_parallel_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    python3 ./mcEA.py -o mcEA/seed_$glass_seed/  --ham fm  --lattice sqr --glass_seed $glass_seed --boundary periodic --L 16 --beta $beta --model half_parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out_EA/seed_$glass_seed/fm_sqr_periodic_L16_beta$beta/nd3_nw4_half_parallel_made_bias_z2_ba0.998_cg1/out_save/6000.state" --k_type global \
    --max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
    done
done
