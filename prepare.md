conda create -p /home/ai-ubuntu/hddnew/Manh/obj_color/env python==3.10 -y
conda activate /home/ai-ubuntu/hddnew/Manh/obj_color/env
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 ultralytics
pip install numpy==1.24.1