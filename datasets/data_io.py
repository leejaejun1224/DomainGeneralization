import numpy as np
import re
import torchvision.transforms as transforms


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_transform_aug():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
    ])

def reshape_image(image):
    w, h = image.size
    processed = get_transform()
    if w <= 1248 and h <= 384:
        top_pad, right_pad = 384 - h, 1248 - w
        assert top_pad > 0 and right_pad > 0
        image = processed(image).numpy()
        image = np.lib.pad(image, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
    else:
        x, y = (w - 1248) // 2, (h - 384) // 2
        image = image.crop((x, y, x + 1248, y + 384))
        image = processed(image).numpy()
    return image

def reshape_disparity(disparity):
    w, h = disparity.shape[1], disparity.shape[0]
    if w <= 1248 and h <= 384:
        top_pad, right_pad = 384 - h, 1248 - w
        assert top_pad > 0 and right_pad > 0
        disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
    else:
        x, y = (w - 1248) // 2, (h - 384) // 2
        disparity = disparity[y:y + 384, x:x + 1248]
    return disparity

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


## pfm 파일 읽어서 numpy array로 바꿈. 
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
