# LMD_reduced_complexity

So far, before the reduction of the models size, we are tryng to understand the effectiveness in our usage case:
  We are using some MNIST digits in-domain and some other OOD.
  For this reason the first edit I made is on the datasets.py

### First results: 
With a train on the first 5 digits of the MNIST and 10 times less iterations (130k), a reconstuction by inpainting of
5 clones with an applied in-domain manifold checkerboard_alt of size 8 mask, we are receiving now a detection of LPIPS AUC: 0.825 (on checkpoint 13)
