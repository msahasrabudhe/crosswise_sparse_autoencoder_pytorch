import  torch
import  torch.utils.data
import  torchvision.utils       as      vutils
import  torchvision.transforms  as      vtransforms

from    attr_dict               import  *
from    utils                   import  *

import  pickle
import  matplotlib.pyplot       as      plt
import  skimage.color           as      skcolour
from    PIL                     import  Image
import  random
import  numpy                   as      np
import  os
import  sys

# Need h5py for HDF5 files. 
import  h5py

# Use short name for torch.nn.functional
F           = torch.nn.functional


# ======================================================================================================
#   Image loader that uses PIL. 
def pil_loader(path):
    """
    pil_loader ::: Uses PIL to read images. 
    """
    img = Image.open(path)
    return img
# ======================================================================================================


# ======================================================================================================
def NormaliseAndFixImageSize(patch_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    NormaliseAndFixImageSize ::: Normalises a PIL image according to given mean and std. 
    Also fixes image size so that it measures patch_size x patch_size. 
    """
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             FixImageSize(patch_size, pad_value=0),
           ])
# ======================================================================================================


# ======================================================================================================
def ColourTransform(mean=1.0, std=0.03):
    """
    ColourTransform ::: Adds data augmentation by breaking the image apart 
    into H & E stains, and randomly modifying their concentration. 
    """
    def transform(img):
        hed         = skcolour.rgb2hed(img / 255.0)
        alphas      = np.random.normal(size=(1,1,3), loc=mean, scale=std)
        hed         = hed * alphas
        img         = skcolour.hed2rgb(hed).astype(np.float32)
        return img

    return transform
# ======================================================================================================


# ======================================================================================================
class RandomRotation(object):
    """
    A randomly chosen rotation is applied to a PIL image.
    """
    def __init__(self, angles_list):
        self.angles_list    = angles_list
    def __call__(self, img):
        A                   = np.random.choice(self.angles_list)
        return img.rotate(A)
# ======================================================================================================


# ======================================================================================================
class FixImageSize(object):
    """
    FixImageSize ::: A transformation to pad an image so that every 
    image can be ensured to be of the same size. The padding is done 
    so that the patch rests in the top-left corner. Uses 
    the nn.functional library. 
    """
    def __init__(self, size, pad_value=0):
        if isinstance(size, int):
            self.size       = (size, size)
        elif isinstance(size, list) and len(size) == 2 and all([isinstance(s, int) for s in size]):
            self.size       = size
        else:
            raise ValueError('Expected either an integer or a list of two integers as the first argument to FixImageSize. Got {}'.format(size))

        self.pad_value      = pad_value

    def __call__(self, image):
        size_x              = image.size(2)
        size_y              = image.size(1)

        pad_list            = [0, 0, 0, 0]
        if size_x < self.size[1]:
            pad_list[1]     = self.size[1] - size_x
        if size_y < self.size[0]:
            pad_list[3]     = self.size[0] - size_y
        
        image               = F.pad(image, pad_list, 'constant', self.pad_value)
        return image
# ======================================================================================================

# ======================================================================================================
#   Dataset for PatchCamelyon images. 
class PatchCamelyonImageset(torch.utils.data.Dataset):
    """
    PatchCamelyonImageset ::: torch.utils.data.Dataset for the
    PatchCamelyon images. 

    Inputs:
        options             AttrDict read from a .yaml configuration file.
        splits              List of splits to include. 
        img_transforms      List of custom transforms to apply. 
                            Normalisation and appropriate padding are
                            included by default.

    The PatchCamelyon dataset can be obtained here: https://github.com/basveeling/pcam
    """
    def __init__(self, options, splits=['train'], img_transforms=None):
        super(PatchCamelyonImageset, self).__init__()

        # Location where .h5 files can be found. 
        self.dataroot       = options.dataroot

        # Ideally, splits should contain only one member.  
        # However, if 'train' is found in splits, the object assumes
        #   this is will be used as a training set. 
        # Colour augmentation is applied in this case. 
        self.if_train       = 'train' in splits

        x_file              = 'camelyonpatch_level_2_split_%s_x.h5' %(splits[0])
        y_file              = 'camelyonpatch_level_2_split_%s_y.h5' %(splits[0])

        for f_ in [x_file, y_file]:
            ff_             = os.path.join(self.dataroot, f_)
            assert(os.path.exists(ff_)), 'File %s must be present in specified dataroot %s.' %(f_, self.dataroot)

        x_file              = os.path.join(self.dataroot, x_file)
        y_file              = os.path.join(self.dataroot, y_file)

        # If patch_size is -1, we use entire images. Otherwise, we extract patches from the images. 
        if options.patch_size == -1:
            self.patch_size = options.image_size
        else:
            self.patch_size = options.patch_size

        # Read files. 
        fp_x                = h5py.File(x_file, 'r')
        self.X              = fp_x['x']
        fp_y                = h5py.File(y_file, 'r')
        self.y              = fp_y['y']

        # Create transforms. 
        self.img_transforms = img_transforms
        self.colour_transform = ColourTransform(mean=1.0, std=0.03)
        self.normalise      = NormaliseAndFixImageSize(self.patch_size, mean=options.pixel_means, std=options.pixel_stds)

    def __getitem__(self, index):
        # Retrieve image at the location index. 
        img                 = Image.fromarray(self.X[index,:,:,:])
        label               = self.y[index].squeeze().astype(np.int64)

        if self.img_transforms is not None:
            img             = self.img_transforms(img)

        # Convert to Numpy array from PIL Image. 
        img                 = np.array(img)

        if self.if_train:
            img             = self.colour_transform(img)

        img                 = self.normalise(img)

        # The dataset also returns the label. However, it is not used by the autoencoder. 
        return img, label

    def __len__(self):
        return self.X.shape[0]
# ======================================================================================================
