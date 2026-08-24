import albumentations as A

augment = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.HorizontalFlip(p=0.5),
])

def augment_image(img):
    return augment(image=img)["image"]