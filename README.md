# Various GANs Analysis

## Datasets Used for each Model
- ### DCGAN
  [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  To get and set up the dataset
  ```console
  $ cd dataset/dog_data
  $ wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
  $ tar xvf images.tar
  $ rm images.tar
  $ mv ./Images/*
  $ rm -r Images
  ```