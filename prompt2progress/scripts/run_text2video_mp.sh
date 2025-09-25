name="test"

ckpt='../checkpoints/base_512_v2'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="../prompts/test_prompts_mp.txt"
res_dir="results"

python3 scripts/evaluation/inference_mp.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 16 \
--frames 32 \
