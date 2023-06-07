conda create -n 3d-cinemagraphy python=3.9
conda activate 3d-cinemagraphy
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install tensorboard loguru scipy lpips tqdm pyyaml ninja opencv-python py-lz4framed av matplotlib scikit-learn
pip install kornia imageio imageio-ffmpeg scikit-image timm
pip install labelme