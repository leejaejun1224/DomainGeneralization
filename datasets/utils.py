import re
import numpy as np
from PIL import Image

def pfm_imread(filename):
    with open(filename, 'rb') as file:
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
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

def pfm_to_png(pfm_path, png_path):
    data, _ = pfm_imread(pfm_path)

    # Normalize to 0–255 and convert to uint8
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    # If grayscale, convert accordingly
    if len(data.shape) == 2:
        img = Image.fromarray(data, mode='L')
    else:
        img = Image.fromarray(data, mode='RGB')

    img.save(png_path)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    pfm_to_png("example.pfm", "output.png")