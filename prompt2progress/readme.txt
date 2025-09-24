## Setup
1. Install environments via anaconda
conda create -n myenv python=3.8.5
conda activate myenv
pip install -r requirements.txt

2. Download pretrained T2V VideoCrafter2 weights, and put the model.ckpt in checkpoints/base_512_v2/model.ckpt.

3. Run the following command:
sh scripts/run_text2video_mp.sh