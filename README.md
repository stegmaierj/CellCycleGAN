# CellCycleGAN
Data synthesis framework for 2D microscopy images using statistical shape models and GANs.


## Ground Truth Preparation
1. Download the supplementary material of Zhong et al., 2012 from https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.2046/MediaObjects/41592_2012_BFnmeth2046_MOESM257_ESM.zip and extract it to a destination of your choice.
2. Copy the files `ExtractImageData.m` and `SegmentCenterNucleus.m` to the extracted `%SI_ZIP_CONTENT%/code/` folder, create a new directory for the training images and execute the file `ExtractImageData.m`. You'll be asked for an output folder, where you can specify the previously created one.
3. The images from the archive will be processed and written to a `*.h5` format that will be used for CNN training.


## Generation of Synthetic Image Data




## References
[1] Zhong, Q., Busetto, A. G., Fededa, J. P., Buhmann, J. M., & Gerlich, D. W. (2012). Unsupervised modeling of cell morphology dynamics for time-lapse microscopy. Nature Methods, 9(7), 711-713.