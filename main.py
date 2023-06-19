# -*- coding:utf-8 -*-
# @Author: IEeya
from myOptimize.prepare import prepare_candidates_and_intersections
from sketch.get_sketch import get_sketch_from_image
from sketch.preload_sketch import preload_sketch
from tools.tools_visualization import visualize_polyscope
import time

def symmetric_build_from_2D_to_3D():

    time_start = time.time()  # 开始计时
    filepath = "./data/student9_house_view1_concept.svg"
    sketch = get_sketch_from_image(filepath)
    cam, sketch = preload_sketch(sketch)
    answer = prepare_candidates_and_intersections(cam, sketch)
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')
    visualize_polyscope(answer, cam)



if __name__ == '__main__':
    symmetric_build_from_2D_to_3D()

