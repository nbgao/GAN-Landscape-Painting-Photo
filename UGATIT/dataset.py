import torch.utils.data as data
from PIL import Image
import os

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    images.append(item)
    return images

class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions))
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform is not None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return 'Dataset {}\n\
                    Number of datapoints: {}\n\
                    Root Location: {}\n\
                    Transforms (if any):{}\n\
                    Target Transforms (if any):{}\n\
                '.format(self.__class__.__name__, 
                         self.__len__(),
                         self.root,
                         self.transform.__repr__(),
                         self.target_transform.__repr__())

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        self.imgs = self.samples


