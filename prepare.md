// spec: gpu support highest CUDA 11.3
// for my GPU CUDA support highest 11.3, go on previous torch local
// u can install torch based on ur CUDA support version , visit torch website and previous torch website 

conda create -p /home/ai-ubuntu/hddnew/Manh/obj_color/env python==3.10 -y
conda activate /home/ai-ubuntu/hddnew/Manh/obj_color/env
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 ultralytics
pip install numpy==1.24.1
pip install fastapi uvicorn pydantic python-multipart


window 
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia 
pip install numpy==1.24.1
pip install fastapi uvicorn pydantic python-multipart

