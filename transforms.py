from torchvision import transforms



def get_data_transforms():
    """
    Returns a single transform pipeline for training, validation, and testing datasets.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])

    return transform

