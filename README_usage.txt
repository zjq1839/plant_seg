Usage:

1) Install dependencies (ideally in a virtualenv):
   pip install -r requirements.txt

2) Prepare a dataset folder with structure:
   data/voc_like/
     images/
       train/*.jpg|png
       val/*.jpg|png
     masks/
       train/*.png  # label map per pixel, 0..num_seen-1 for seen, others as background/unseen
       val/*.png

3) Edit configs/voc.yaml to set num_seen and dataset root.

4) Train:
   python train.py --config configs/voc.yaml

5) Inference on one image:
   python infer.py --config configs/voc.yaml --ckpt checkpoints/best.pth --image path/to/image.jpg --out pred.png