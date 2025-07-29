from .kitti2012 import KITTI2012Dataset
from .kitti2015 import KITTI2015Dataset
from .cityscapes import CityscapesDataset
from .driving_stereo import DrivingStereoDataset
# from .sceneflow import FlyingThingDataset
from .sceneflow_dataset_augmentation import FlyingThingDataset

# ㅋㅋ 이거 위에 파이썬 이름에 underbar있으면 안됨

__datasets__ = {
    "cityscapes": CityscapesDataset,
    "kitti_2015": KITTI2015Dataset,
    "kitti_2012": KITTI2012Dataset,
    "flyingthing": FlyingThingDataset,
    "driving_stereo": DrivingStereoDataset
}