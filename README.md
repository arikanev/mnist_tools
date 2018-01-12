## Mnist Tools

# mnist_seg.py
A script for generating the mnist segmentation dataset, mnist segmentations are useful when one requires quick segmentation model convergence, i.e. for hyperparameter testing.

Run with

`python mnist_seg.py --t mask_threshold (int) --s save_path (str) --p pkl_path (str)` and optionally, `--RGB (Bool)`


# mnistmPytorchLoader.py
A script to load mnistm into pytorch using the existing dataloader class. You should use this script as an auxilary to your pytorch model and train.py.

Run with
```
python mnistmPytorchLoader.py
```

# mnistm_gen.py
A script for generating the mnistm dataset, which uses the mnist segmentations generated with **mnist_seg.py** as ground truth.

Run with
```
python mnistm_gen.py --mp mnist_path (str) --ip imgnet_path (str) --s save_path (str)
```
