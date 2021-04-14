"""
Script to build a data generator for the training process
"""

# Import the necessary packages
import numpy as np
import tensorflow as tf
import yaml, os
from tqdm import tqdm
from cv2 import cv2 as cv

# importing augmentation functions
from albumentations import Compose, OneOf
from albumentations import Flip, Rotate, RandomScale, ShiftScaleRotate, RandomCrop
from albumentations import HueSaturationValue, RandomBrightnessContrast
from albumentations import MedianBlur, GaussianBlur, GaussNoise, MotionBlur, ElasticTransform

# Import in-house functions
from ._cvc_clinic_reader import read_cvc_clinic_data
from ._kvasir_seg_reader import read_kvasir_seg_data
from ._core_utils import _load_data_unet
from ._core_utils import _load_data_unet_dice
from ._core_utils import _load_data_fcn8
from ._core_utils import _load_data_deeplabv3
from ._core_utils import _load_data_unet_attn
from ._core_utils import _load_data_unet_guided_attn
from ._core_utils import _load_data_gar_net
from ._core_utils import _load_data_segnet
from ._core_utils import _load_data_dilated_resfcn
from ._core_utils import _load_data_gar_net_exp


# Implement the class data generator powered with tf.data
class DataGenerator:
    """
    A class that implements a tensorflow data pipeline based data generator for loading and training the
    polyp dataset
    """
    # Note: Override built-in functions
    def __init__(self, X_paths: list, y_paths: list, image_size: tuple, model_name: str, batch_size: int,
                 dataset_family: str, initial_size=None, aug_config_path=None, shuffle=True):
        """
        A class that implements a tensorflow data pipeline based data generator for loading and training the
        polyp dataset. Currently supports CVC-Clinic Dataset.
        TODO: more dataset support in progress

        Parameters
        ----------
        X_paths : list
            The path list of all the original input images.
        y_paths : list
            The path list of all the segmentation maps.
        image_size : tuple
            A tuple mentioning the size of the output image as (width, height)
        model_name : str
            The model used for training and testing. This is needed to prepare the data accordingly.
            Currently supported models:
                1. UNet
                2. UNet-Attention
                3. MIMO-Net
                4. AI-Net
        batch_size : int
            The size of the batch to process.
        dataset_family : str
            Mention the family of the dataset. Supported Family:
                1. "CVC-ClinicDB" -- CRAG dataset path
        initial_size : tuple, optional
            A tuple mentioning the initial size of the images to be kept as (W, H). This is useful
            when we want efficient random cropping from the images. eg. initial_size can be (500, 500) out of which we can
            random crop the final input image in the image_size as (464, 464). NOTE: initial_size = image_size if it is
            None. (Default is None)
        aug_config_path : str, optional
            The path of the augmentation configuration file. Mention the configuration file to enable augmentation.
            NOTE: Augmentation is disabled if value is None. (default is None)
        shuffle : bool, optional
            The flag to enable shuffling. (default is None)
        """
        # Get the attributes
        self._X_path = X_paths
        self._y_path = y_paths
        self.image_size = image_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataset_family = dataset_family
        self.initial_size = initial_size if initial_size is not None else image_size
        self.aug_config_path = aug_config_path
        self.shuffle = shuffle

        # Get the dataset
        if dataset_family == "CVC-ClinicDB":
            print("[INFO]: Using CVC-ClinicDB Reading Utilities. Loading code :)")

        elif dataset_family == "Kvasir-Seg":
            print("[INFO]: Using Kvasir-Seg Reading Utilities. Loading code :)")

        # TODO: Implement Etis Dataset support
        else:
            print("[ERROR]: {0} dataset family is not supported!!".format(dataset_family))
            raise NotImplementedError

        # Setup Augmentation
        self.aug_config_path = aug_config_path
        if self.aug_config_path is None:
            print("""[WARN]: Augmentation is disabled. Please mention the augmentation configuration file while calling 
            the revoke_augmentation() function to enable augmentation.
            """)
        else:
            self.setup_augmentation()
            print("[INFO]: Loaded the Augmentation Functions based on the configuration.")
            print("[INFO]: To apply a change, call reload_augmentation()")

        # Load the data
        self._load_data()

        # Get indexes
        self.indexes = np.arange(len(self._dataset_X))

    def __len__(self) -> int:
        """
        Function to return the length of the default steps_per_epoch.

        Returns
        -------
        int
            The steps per epoch plausible by the given batch_size
        """
        steps_per_epoch = len(self.indexes) // self.batch_size
        return steps_per_epoch

    # Setup Augmentation
    # NOTE: Implement augmentation function
    def _read_augmentation_config(self) -> dict:
        """
        Function to read the yaml configuration file and return the dictionary.

        Returns
        -------
        dict
            The dictionary of the configuration file
        """
        if self.aug_config_path is None:
            print("[ERROR]: Augmentation Configuration Path not Mentioned!")
            raise FileNotFoundError
        else:
            return yaml.load(open(self.aug_config_path, mode="r"), Loader=yaml.FullLoader)

    def setup_augmentation(self) -> None:
        """
        Function to setup augmentation functions with the loaded augmentation configuration using the functions of
        albumentation library.

        Returns
        -------
        None
        """
        aug_config = self._read_augmentation_config()

        # Retrieve Configurations
        HSV = aug_config["HSV_Saturation"]
        RBC = aug_config["Random_Brightness_Contrast"]
        Flip_aug = aug_config["Flip"]
        Linear_aug = aug_config["Linear_Augmentation"]
        Median_Blur = aug_config["Median_Blur"]
        Gaussian_Blur = aug_config["Gaussian_Blur"]
        Gaussian_Noise = aug_config["Gaussian_Noise"]
        Motion_Blur = aug_config["Motion_Blur"]
        Elastic_Blur = aug_config["Elastic_Blur"]

        # Load up functions
        self.aug_fn = Compose([
            HueSaturationValue(hue_shift_limit=HSV["hue_shift_limit"], sat_shift_limit=HSV["sat_shift_limit"],
                               val_shift_limit=HSV["val_shift_limit"], always_apply=HSV["always_apply"],
                               p=HSV["p"]),
            RandomBrightnessContrast(brightness_limit=RBC["brightness_limit"], contrast_limit=RBC["contrast_limit"],
                                     brightness_by_max=RBC["brightness_by_max"], always_apply=RBC["always_apply"],
                                     p=RBC["p"]),
            OneOf([
                MedianBlur(Median_Blur["blur_limit"], Median_Blur["always_apply"], Median_Blur["p"]),
                GaussianBlur(Gaussian_Blur["blur_limit"], Gaussian_Blur["always_apply"], Gaussian_Blur["p"]),
                MotionBlur(Motion_Blur["blur_limit"], Motion_Blur["always_apply"], Motion_Blur["p"]),
            ], p=1.0),
            GaussNoise((Gaussian_Noise["var_limit_low"], Gaussian_Noise["var_limit_high"]), Gaussian_Noise["mean"],
                       Gaussian_Noise["always_apply"], Gaussian_Noise["p"]),
        ])

        self.aug_fn_all = Compose([
            Flip(always_apply=Flip_aug["always_apply"], p=Flip_aug["p"]),
            ShiftScaleRotate(rotate_limit=90, shift_limit=0.0625, interpolation=cv.INTER_CUBIC,
                             border_mode=cv.BORDER_CONSTANT, value=0, always_apply=Linear_aug["always_apply"],
                             p=Linear_aug["p"]),
            RandomCrop(height=self.image_size[1], width=self.image_size[0], always_apply=True, p=1.0),
        ])

        return None

    def reload_augmentation(self, aug_config_path: str):
        """
        Function to reload the augmentation functions based on the new configuration file if mentioned.

        Parameters
        ----------
        aug_config_path : str
            The path of the .yaml configuration file of the augmentation functions.

        Returns
        -------
        None
        """
        if type(aug_config_path) == str:
            raise TypeError
        if not os.path.exists(aug_config_path):
            raise FileNotFoundError

        # Augmentation
        print("[INFO]: Generating augmentation functions")
        self.setup_augmentation()
        return None

    def _apply_augmentation(self, img, seg):
        """
        Function to apply augmentation, and return the tuple of (image, seg_map) for training the model.

        Parameters
        ----------
        img : np.ndarray
            The image input numpy array.
        seg : np.ndarray
            The segmentation input numpy array.

        Returns
        -------
        tuple
            A tuple of (image, seg_map)
        """
        # Image and segmentation
        img = tf.cast(img, dtype=tf.uint8)
        seg = tf.cast(seg, dtype=tf.uint8)

        # Get the numpy version
        img = img.numpy().copy()
        seg = seg.numpy().copy()

        if self.aug_config_path is not None:
            # Apply augmentations
            transformed = self.aug_fn(image=img)
            img = transformed["image"]

            transformed = self.aug_fn_all(image=img, mask=seg)
            img = transformed["image"]
            seg = transformed["mask"]

        elif self.image_size != self.initial_size:
            img = cv.resize(img, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_CUBIC)
            seg = cv.resize(seg, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_CUBIC)

        else:
            None

        seg[seg >= 128] = 255
        seg[seg < 128] = 0

        return img, seg

    # Setup image loading in RAM
    def _load_data(self) -> None:
        """
        Function to load all the images in the RAM.

        Returns
        -------
        None
        """
        self._dataset_X = []
        self._dataset_y = []

        # Iterate through the path
        print("[INFO]: Loading the images in-memory")
        with tqdm(total=len(self._dataset_X)) as pbar:
            for image_path, seg_path in zip(self._X_path, self._y_path):
                # Read the images
                if self.dataset_family == "CVC-ClinicDB":
                    temp_img, temp_seg = read_cvc_clinic_data(image_path, seg_path, resize=self.initial_size)

                elif self.dataset_family == "Kvasir-Seg":
                    temp_img, temp_seg = read_kvasir_seg_data(image_path, seg_path, resize=self.initial_size)

                else:
                    raise NotImplementedError

                # Load the X data
                self._dataset_X.append(temp_img)
                # load the y data
                self._dataset_y.append(temp_seg)
                # Update progress
                pbar.update(1)

        return None

    # NOTE: IMPLEMENT REFINERY FUNCTIONALITY
    def refine_data_unet(self, img, wmap, seg):
        """
        Function to refine the data for the tf.data pipeline for the original U-Net

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        wmap : tf.Tensor
            The weight map for the given image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of ((img, wmap), seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        wmap.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 2)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 2)))

        return (img, wmap), seg

    def refine_data_unet_dice(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the given U-Net dice model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        return img, seg

    def refine_data_fcn8(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the given FCN8 model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        return img, seg

    def refine_data_deeplabv3(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the given deeplabv3 model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        return img, seg

    def refine_data_unet_attn(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the given UNet Attention model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        return img, seg

    def refine_data_unet_guided_attn(self, img, seg, attn1, attn2, attn3, attn4):
        """
        Function to refine the data for the tf.data pipeline for the given UNet Guided Attention model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample
        attn1 : tf.Tensor
            The expected Attention map for the 1st level
        attn2 : tf.Tensor
            The expected Attention map for the 2nd level
        attn3 : tf.Tensor
            The expected Attention map for the 3rd level
        attn4 : tf.Tensor
            The expected Attention map for the 4th level

        Returns
        -------
        tuple
            A tuple of (img, (seg, attn1, attn2, attn3, attn4))
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))
        attn1.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))
        attn2.set_shape(tf.TensorShape((self.image_size[0] // 2, self.image_size[1] // 2, 1)))
        attn3.set_shape(tf.TensorShape((self.image_size[0] // 4, self.image_size[1] // 4, 1)))
        attn4.set_shape(tf.TensorShape((self.image_size[0] // 8, self.image_size[1] // 8, 1)))

        return img, (seg, attn1, attn2, attn3, attn4)

    def refine_data_segnet(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the given SegNet model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        return img, seg

    def refine_data_gar_net_exp(self, img, seg, attn1, attn2, attn3, attn4):
        """
        Function to refine the data for the tf.data pipeline for the given GAR-Net model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample
        attn1 : tf.Tensor
            The attention map 1 for the given sample
        attn2 : tf.Tensor
            The attention map 2 for the given sample
        attn3 : tf.Tensor
            The attention map 3 for the given sample
        attn4 : tf.Tensor
            The attention map 4 for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, (seg, attn1, attn2, attn3, attn4))
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        attn1.set_shape(tf.TensorShape((self.image_size[0] // 2, self.image_size[1] // 2, 1)))
        attn2.set_shape(tf.TensorShape((self.image_size[0] // 4, self.image_size[1] // 4, 1)))
        attn3.set_shape(tf.TensorShape((self.image_size[0] // 8, self.image_size[1] // 8, 1)))
        attn4.set_shape(tf.TensorShape((self.image_size[0] // 16, self.image_size[1] // 16, 1)))

        return img, (seg, attn1, attn2, attn3, attn4)

    def refine_data_gar_net(self, img, seg, attn1, attn2, attn3, attn4):
        """
        Function to refine the data for the tf.data pipeline for the given GAR-Net model

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample
        attn1 : tf.Tensor
            The attention map 1 for the given sample
        attn2 : tf.Tensor
            The attention map 2 for the given sample
        attn3 : tf.Tensor
            The attention map 3 for the given sample
        attn4 : tf.Tensor
            The attention map 4 for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, (seg, attn1, attn2, attn3, attn4))
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))

        attn1.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 1)))
        attn2.set_shape(tf.TensorShape((self.image_size[0] // 2, self.image_size[1] // 2, 1)))
        attn3.set_shape(tf.TensorShape((self.image_size[0] // 4, self.image_size[1] // 4, 1)))
        attn4.set_shape(tf.TensorShape((self.image_size[0] // 8, self.image_size[1] // 8, 1)))

        return img, (seg, attn1, attn2, attn3, attn4)

    def refine_data_dilated_resfcn(self, img, seg):
        """
        Function to refine the data for the tf.data pipeline for the Dilated ResFCN

        Parameters
        ----------
        img : tf.Tensor
            The tensor image
        seg : tf.Tensor
            The segmentation map for the given sample

        Returns
        -------
        tuple
            The refined tuple of (img, seg)
        """
        # Set the shape
        img.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 3)))
        seg.set_shape(tf.TensorShape((self.image_size[0], self.image_size[1], 2)))

        return img, seg

    # NOTE: IMPLEMENT TF FUNCTIONALITY
    def generate_data(self):
        """
        Function to generate indefinite amount of data. Shuffles each time if shuffle is enabled. Returns the (img, seg)
        from the loaded dataset.

        Returns
        -------
        tuple
            A tuple of (img, seg) from the images.
        """
        # Shuffle
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

        while True:
            for index in self.indexes:
                # yield the data
                yield self._dataset_X[index], self._dataset_y[index]

            # Shuffle
            if self.shuffle is True:
                np.random.shuffle(self.indexes)

    def get_tf_data(self):
        """
        Function to get the tf.data.Dataset for the given inputs.

        Returns
        -------
        tf.data.Dataset
            A tf.data.Dataset for the given model to consume for training/testing.
        """
        dataset = tf.data.Dataset.from_generator(self.generate_data,
                                                 output_types=(tf.uint8, tf.uint8),
                                                 output_shapes=(
                                                     tf.TensorShape((self.initial_size[0], self.initial_size[1], 3)),
                                                     tf.TensorShape((self.initial_size[0], self.initial_size[1]))
                                                 ))

        dataset = dataset.map(lambda img, seg: tf.py_function(self._apply_augmentation,
                                                              inp=[img, seg],
                                                              Tout=[tf.float32] * 2),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Handle Different model training
        if self.model_name in ["UNet-Dice", "ResUNet", "ResUNet++", "SE-UNet", "Dilated-UNet"]:
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_unet_dice,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_unet_dice,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "UNet":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_unet,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 3),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_unet,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "FCN8":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_fcn8,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_fcn8,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "SegNet":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_segnet,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_segnet,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "DeepLabv3":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_deeplabv3,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_deeplabv3,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "UNet-Attn":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_unet_attn,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_unet_attn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "UNet-GuidedAttn":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_unet_guided_attn,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 6),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_unet_guided_attn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "Dilated-ResFCN":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_dilated_resfcn,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 2),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_dilated_resfcn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "GAR-Net-Experimental":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_gar_net_exp,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 6),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_gar_net_exp,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif self.model_name == "GAR-Net":
            dataset = dataset.map(lambda img, seg: tf.py_function(_load_data_gar_net,
                                                                  inp=[img, seg],
                                                                  Tout=[tf.float32] * 6),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(self.refine_data_gar_net,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        else:
            raise NotImplementedError

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset.repeat()


