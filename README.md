# voicecraft
## installation 

pull from dockerhub
```bash
# download docker image
enroot import docker://thivux/voicecraft:audiocraft
```

from thivt1's lustre
```bash
cd 

# create docker container
enroot create -n [container-name] thivux+voicecraft+complete_env.sqsh

# run container 
NVIDIA_DISABLE_REQUIRE=1 enroot start --env HOME=/root \
--mount ../code/VoiceCraft:/workspace/VoiceCraft \
--mount ../code/oneshot:/workspace/oneshot \
--mount /home/thivt1/.cache:/home/thivt1/.cache \
--mount /home/thivt1/.conda:/home/thivt1/.conda \
--mount /home/thivt1/data:/home/thivt1/data \
[container-name] /bin/bash

# init conda
source /root/.bashrc
# activate env 
source activate /conda/voicecraft
```

## training 
prepare data 

```bash
cd ./data

python phonemize_encodec_encode_hf.py \
--dataset_size xs \
--download_to /home/thivt1/data/datasets/gigaspeech_xs \
--save_dir /home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest \
--encodec_model_path ../pretrained_models/encodec_4cb2048_giga.th \
--mega_batch_size 120 \
--batch_size 32 \
--max_len 30000
```

train 
```bash
cd ./z_scripts
bash e830M.sh
```
