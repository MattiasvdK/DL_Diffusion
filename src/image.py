from torchvision.transforms.v2 import Lambda, ToTensor, Compose, Resize, RandomCrop, CenterCrop, ToPILImage

class ImageTransformer():
    def __init__(self, image_size):
        self.image_size = image_size
        self.transform = Compose([
            Resize(image_size),
            RandomCrop(image_size),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1),
        ])
        self.inverse_transform = Compose([
            Lambda(lambda x: (x + 1) / 2),
            ToPILImage(),
        ])
        
    def __call__(self, x):
        return self.transform(x)
    
    def inverse(self, x):
        return self.inverse_transform(x)
    
