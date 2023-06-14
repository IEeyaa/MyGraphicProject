# -*- coding:utf-8 -*-
# @Author: IEeya
from sketch.get_sketch import get_sketch_from_image
from sketch.preload_sketch import preload_sketch, visualize_lines
from sketch.symmetric_drive import symmetric_driven_build
from tools_drawing.visualization import sketch_plot, visualize_polyscope


def symmetric_build_from_2D_to_3D():
    filepath = "./data/sketch.svg"
    sketch = get_sketch_from_image(filepath)
    # sketch_plot(sketch)
    cam, sketch = preload_sketch(sketch)
    answer = symmetric_driven_build(cam, sketch)
    visualize_polyscope(answer, cam)
    # 可视化
    # visualize_lines(sketch)


if __name__ == '__main__':
    symmetric_build_from_2D_to_3D()

