from .kitti2012 import KITTI2012Dataset
from .kitti2015 import KITTI2015Dataset
from .cityscapes import CityscapesDataset
from .driving_stereo import DrivingStereoDataset

__datasets__ = {
    "cityscapes": CityscapesDataset,
    "kitti_2015": KITTI2015Dataset,
    "kitti_2012": KITTI2012Dataset,
    "driving_stereo": DrivingStereoDataset
}