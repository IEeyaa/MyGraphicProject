import numpy as np

def trimStraightStrokeCoordinatesDrawingField(stroke, sketch_height):
    
    points = np.array([p.coords for p in stroke.points_list])
    # Evaluate which strokes points are inside the sketching area:
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    
    x_vals, y_vals, inside = trimLineCoordinatesDrawingField(x_vals, y_vals, sketch_height)

    # Keep only strokes inside the sketching area:
    stroke.points_list = [p for p_id, p in 
        enumerate(stroke.points_list) if p_id in inside]

def interpolate_x(seg, val_new):
    y2 = seg[1,1]
    y1 = seg[0,1]
    
    x2 = seg[1,0]
    x1 = seg[0,0]
    
    dy = y2 - y1
    y_grows = dy > 0
    dx = x2 - x1
    
    r = dy/dx
    
    if y_grows:
        coord = y1 + r*(val_new - x1)
    else:
        coord = y2 + r*(val_new - x2)
    return coord

def interpolate_y(seg, val_new):
    y2 = seg[1,1]
    y1 = seg[0,1]
    
    x2 = seg[1,0]
    x1 = seg[0,0]
    
    dx = x2 - x1
    dy = y2 - y1
    x_grows = dx > 0
    
    r = dx/dy
    
    if x_grows:
        coord = x1 + r*(val_new - y1)
    else:
        coord = x2 + r*(val_new - y2)
    return coord


def trimLineCoordinatesDrawingField(x_vals, y_vals, sketch_height):
    
    min_b = 0
    max_b = sketch_height

    # Determine strokes directions:
    
    grows_x = (x_vals[-1] - x_vals[0]) > 0
    grows_y = (y_vals[-1] - y_vals[0]) > 0
    
    inds_out = np.argwhere(x_vals < min_b).flatten()
    x_vals, y_vals = updateOneClipCoordinateMin(x_vals, y_vals, inds_out, grows_x, 'x', min_b)
    # x - height
    inds_out = np.argwhere(x_vals > max_b).flatten()
    x_vals, y_vals = updateOneClipCoordinateMax(x_vals, y_vals, inds_out, grows_x, 'x', max_b)
    # y 0 
    inds_out = np.argwhere(y_vals < min_b).flatten()
    x_vals, y_vals = updateOneClipCoordinateMin(x_vals, y_vals, inds_out, grows_y, 'y', min_b)
    # y - height
    inds_out = np.argwhere(y_vals > max_b).flatten()
    x_vals, y_vals = updateOneClipCoordinateMax(x_vals, y_vals, inds_out, grows_y, 'y', max_b)
    
    # Define points inside:
    inside = np.argwhere(  np.logical_and(x_vals >= min_b,
                    np.logical_and(x_vals <= max_b,
                    np.logical_and(y_vals >= min_b,
                        y_vals <= max_b)))).flatten()

    # Keep only strokes inside the sketching area:
    x_vals = x_vals[inside]
    y_vals = y_vals[inside]

    return x_vals, y_vals, inside

def updateOneClipCoordinateMin(x_vals, y_vals, inds_out, grows_dim, interp_coord, min_b):

    num_points = len(y_vals)
    update = False
    if len(inds_out) > 0:
        if grows_dim:
            vert_num = inds_out[-1]

            if vert_num != num_points-1:
                update = True
                seg = np.array([[x_vals[vert_num], y_vals[vert_num]],
                                [x_vals[vert_num+1], y_vals[vert_num+1]]])
        else:
            vert_num = inds_out[0]

            if vert_num != 0:
                update = True
                seg = np.array([[x_vals[vert_num], y_vals[vert_num]],
                                [x_vals[vert_num-1], y_vals[vert_num-1]]])

        if update:
            if interp_coord == 'y':
                x_vals[vert_num] = interpolate_y(seg, min_b)
                y_vals[vert_num] = min_b
            else:
                x_vals[vert_num] = min_b
                y_vals[vert_num] = interpolate_x(seg, min_b)

    return x_vals, y_vals

def updateOneClipCoordinateMax(x_vals, y_vals, inds_out, grows_dim, interp_coord, max_b):
                                
    num_points = len(y_vals)
    update = False
    if len(inds_out) > 0:
        if grows_dim:
            vert_num = inds_out[0]

            if vert_num != 0:
                update = True
                seg = np.array([[x_vals[vert_num], y_vals[vert_num]],
                                [x_vals[vert_num-1], y_vals[vert_num-1]]])
        else:
            vert_num = inds_out[0]

            if vert_num != num_points-1:
                update = True
                seg = np.array([[x_vals[vert_num], y_vals[vert_num]],
                                [x_vals[vert_num+1], y_vals[vert_num+1]]])

        if update:
            if interp_coord == 'y':
                x_vals[vert_num] = interpolate_y(seg, max_b)
                y_vals[vert_num] = max_b
            else:
                x_vals[vert_num] = max_b
                y_vals[vert_num] = interpolate_x(seg, max_b)
    return x_vals, y_vals