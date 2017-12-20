## Mnist Tools

# mnist_seg.py
A script for generating an mnist segmentation dataset, mnist segmentations are useful when one requires quick segmentation model convergence, i.e. for hyperparameter testing.

Run with
```
python mnist_seg.py --t mask_threshold (int) --s save_path (str) --p pkl_path (str) --RGB
```

# mnistmPytorchLoader.py
A script to load mnistm into pytorch using the existing dataloader class. You should use this script as an auxilary to your pytorch model and train.py.

Run with
```
python mnistmPytorchLoader.py
```
