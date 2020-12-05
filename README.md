# Various GANs Analysis

:warning:As for training any GAN, please pay attention to the followings to stablize the training! :warning:
- Check that neither model has "won". If either the generator-gan-loss or the discriminator-loss gets very low, it's an indicator that one model is dominating the other, and you are not successfully training the combined model.
- The value log(2) = 0.69 is a good reference point for these losses, as it indicates a perplexity of 2: That the discriminator is on average equally uncertain about the two options.
- For the discriminator loss, a value below 0.69 means the discriminator is doing better than random on the combined set of real+generated images.
- For the generator-gan-loss, a value below 0.69 means the generator is doing better than random at foolding the descriminator.

## Datasets Used for each Model
- ### For DCGAN, WGAN, WGAN-GP
  [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)<br>
  To get and set up the dataset
  ```console
  $ cd dataset/dogs_data
  $ wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
  $ tar xvf images.tar
  $ rm images.tar
  ```
  
## Results
- ### DCGAN
  <div align="center">
  <img src="https://user-images.githubusercontent.com/37681936/75147367-7a52ed80-5740-11ea-88f6-445813b33b88.png" alt="DCGAN Result">
  </div>
- ### WGAN
  <div align="center">
  <img src="https://user-images.githubusercontent.com/37681936/74636528-88869400-51ab-11ea-983a-146934353f8b.png" alt="WGAN Result">
  </div>
- ### WGAN-GP
  <div align="center">
  <img src="https://user-images.githubusercontent.com/37681936/75090485-ad617980-55a6-11ea-88e3-3639ab2bc1e4.png" alt="WGAN Result">
  </div>

- ### For Pix2Pix
  [pix2pix datasets](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)
  
## References
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
