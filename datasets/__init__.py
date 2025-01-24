from .kitti2012 import KITTI2012Dataset
from .kitti2015 import KITTI2015Dataset
from .cityscapes import CityscapesDataset


__datasets__ = {
    "cityscapes": CityscapesDataset,
    "kitti2015": KITTI2015Dataset,
    "kitti2012": KITTI2012Dataset
}