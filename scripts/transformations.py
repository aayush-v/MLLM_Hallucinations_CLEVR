import albumentations as A
import cv2
import random
import inspect

random.seed(7777)
# random.seed(12321)
# random.seed(10101)


def pickMethod(img_path):
    # methods = [gaussianLowPassFilter, medianFilter, zoomBlur, gaussNoiseAddition, isoNoiseAddition, multiplicativeNoiseAddition, randomGammaAddition, randomBrightnessContrastAddition, imageCompression]
    methods = [gaussianLowPassFilter, medianFilter, zoomBlur, gaussNoiseAddition, isoNoiseAddition, multiplicativeNoiseAddition]
    method_picked = random.choice(methods)
    method_default_params = inspect.signature(method_picked)
    return method_picked(img_path), method_picked.__name__ + str(inspect.signature(method_picked)) 

def gaussianLowPassFilter(img_path, blur_limit=(21, 21), sigma_limit=(21, 21)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def medianFilter(img_path, blur_limit=21):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.MedianBlur(blur_limit=blur_limit, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def zoomBlur(img_path, max_factor=1.31, step_factor=(0.01, 0.03)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.ZoomBlur(max_factor=max_factor, step_factor=step_factor, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def gaussNoiseAddition(img_path, var_limit=(10.0, 50.0), mean=0, per_channel=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def isoNoiseAddition(img_path, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.ISONoise(color_shift=color_shift, intensity=intensity, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def multiplicativeNoiseAddition(img_path, multiplier=(0.9, 1.1), per_channel=False, elementwise=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.MultiplicativeNoise(multiplier=multiplier, per_channel=per_channel, elementwise=elementwise, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img
    
def randomGammaAddition(img_path, gamma_limit=(80, 140)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.RandomGamma(gamma_limit=gamma_limit, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def randomBrightnessContrastAddition(img_path, brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img

def imageCompression(img_path, quality_lower=20, quality_upper=50):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=1)], p=1)
    transformed = transform(image=img)
    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)

    return aug_img
