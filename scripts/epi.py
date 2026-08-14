import struct

import numpy as np
from matplotlib import pyplot as plt


def read_epi_content_file(file_path: str, display=False):
    """
    Reads an epi.content file and returns its content
    :param file_path: the path to the .epi.content file to open
    :param display: if the function should display the image in runtime
    :return: a tuple with a numpy array of the image data, and the boundaries in mm
    """
    with open(file_path, 'rb') as file:
        nb_cols, nb_rows = struct.unpack('2i', file.read(8))

        x_min, y_min, x_max, y_max = struct.unpack('4f', file.read(16))

        pixel_count = nb_cols * nb_rows
        pixel_array_format = f'{pixel_count}f'
        pixel_array = struct.unpack(pixel_array_format, file.read(4 * pixel_count))

        img = np.reshape(pixel_array, (nb_rows, nb_cols))

        if display:
            plt.imshow(img, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='jet')
            plt.colorbar(label='Pixel value')
            plt.title('Image from raw file')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    return img, [x_min, y_min, x_max, y_max]