# -*- coding:utf-8 -*-
# @Author: IEeya
from myOptimize.prepare import prepare_candidates_and_intersections
from sketch.get_sketch import get_sketch_from_image
from sketch.preload_sketch import preload_sketch
from tools.visualization import visualize_polyscope


def symmetric_build_from_2D_to_3D():
    filepath = "./data/sketch.svg"
    sketch = get_sketch_from_image(filepath)
    cam, sketch = preload_sketch(sketch)
    answer = prepare_candidates_and_intersections(cam, sketch)
    visualize_polyscope(answer, cam)


if __name__ == '__main__':
    symmetric_build_from_2D_to_3D()

