# LMD_reduced_complexity

So far, before the reduction of the models size, we are tryng to understand the effectiveness in our usage case:
  We are using some MNIST digits in-domain and some other OOD.
  For this reason the first edit I made is on the datasets.py

### First results: 
With a train on the first 5 digits of the MNIST and 10 times less iterations (130k), a reconstuction by inpainting of
5 clones with an applied in-domain manifold checkerboard_alt of size 8 mask, we are receiving now a detection of LPIPS AUC: 0.825 (on checkpoint 13)


### Reduced model-> 93 MB params found in ddpm_mnist.py (subvp probability flow)
MNIST vs KMNIST-> 0.9501 LPIPS


## Model comparison with mnist digits 0,1,2,3 IN_domain and 4,5,9 Out_of_domain

  105 MB: LPIPS ROC AUC: 0.9942
  20 MB: LPIPS ROC AUC: 0.9560
  12 MB: LPIPS ROC AUC: 0.9408
  
