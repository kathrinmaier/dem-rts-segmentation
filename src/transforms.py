import kornia.augmentation as K


def augmentations() -> K.AugmentationSequential:
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(degrees=(0, 90), p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        data_keys=["image", "mask"],
    )