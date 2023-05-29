# -*- coding:utf-8 -*-
# @Author: IEeya
from get_sketch import get_sketch_from_image
from preload_sketch import preload_sketch


def symmetric_build_from_2D_to_3D():
    filepath = "./data/sketch.svg"
    sketch = get_sketch_from_image(filepath)
    preload_sketch(sketch)


if __name__ == '__main__':
    symmetric_build_from_2D_to_3D()

