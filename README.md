# Style transfer
## Dependencies
* *cv2*
* *numpy*
* *skimage*
* *tensorflow*
* *jupyter notebook*
* [VGG19 network:](https://github.com/machrisaa/tensorflow-vgg) I`m using VGG19 NPY with removed FC layers, it decreased size of weight matrix from  561,203KB to 118,308KB, vgg19_fc_less.npy plased in this repository in rar file.

## ================================================
  This is my first implemantation of image style/content transfer, based on [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) with few improvements suggested by [Improving the Neural Algorithm of Artistic Style](https://arxiv.org/abs/1605.04603) such that used shifted activations when computing Gram matrices.

* `-another row`: noice
