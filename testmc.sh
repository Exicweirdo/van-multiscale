L=16
for beta in `seq 0.42 0.03 0.48`; do
python3 ./mc2d.py -o out_mc --ham fm --lattice sqr --L $L --beta $beta --model parallel_made --net_depth 3 --net_width 4 --bias --z2 --beta_anneal 0.998 --clip_grad 1 --cuda 0 --checkpoint "out/fm_sqr_periodic_L${L}_beta$beta/nd3_nw4_parallel_made_bias_z2_ba0.998_cg1/out_save/9000.state" --k_type global \
--max_step 20000 --print_step 1000 --save_step 1000 --visual_step 1000
done