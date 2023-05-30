import numpy as np


def findFirstVisibleStrokeTowardsVP(probabilitiesVP, sketch):
    ind_stroke = 0  # Index of the stroke in the sketch
    lj = 0  # Index of the stroke among consecutive straight strokes

    lines_group = np.argmax(probabilitiesVP, axis=1)

    while sketch.strokes[ind_stroke].line_group != 0 or lines_group[lj] == 3:
        if lines_group[lj] == 3:
            lj += 1
        ind_stroke += 1

    return ind_stroke


def findXYZ(vp):
    # Vanishing points on a new order:
    vp_ = np.zeros([3, 2])
    xyz_inds = np.zeros(3)

    # z vanishing point:
    vals = np.argsort(vp[:, 0])
    z_pos = vals[1]

    # Vertical direction
    vp_[2, :] = vp[z_pos, :]
    xyz_inds[2] = z_pos

    # x,y vanishing points:
    inds = vals[[0, 2]]
    left_vp = np.argmax(vp_[2, 0] - vp[inds, 0])
    right_vp = np.argmin(vp_[2, 0] - vp[inds, 0])

    left_vp = inds[left_vp]
    right_vp = inds[right_vp]
    mean_y = np.mean(vp[[left_vp, right_vp], 1])

    ZUP = vp[z_pos, 1] < mean_y

    # Rigth coordinate system:
    if ZUP:
        # z is up
        # left vp is 'y' and right is 'x'
        vp_[1, :] = vp[left_vp, :]
        xyz_inds[1] = left_vp
        vp_[0, :] = vp[right_vp, :]
        xyz_inds[0] = right_vp
    else:
        # z is down
        # left vp is 'x' and right is 'y'
        vp_[0, :] = vp[left_vp, :]
        xyz_inds[0] = left_vp
        vp_[1, :] = vp[right_vp, :]
        xyz_inds[1] = right_vp

    return vp_, xyz_inds


def checkInfinityConditionVP(vp, width):
    w = width
    h = width

    return np.logical_or(vp[:, 0] > 10 * w, vp[:, 1] > 10 * h)


def getPrincipalPoint(vp, mask_infinite_vps):
    # Input:
    #   vp = 3 by 2 array of image cordinate of vanishign points. [x1 y1; x2
    #   y2; x3 y3];
    # 
    # Output:
    #   [x y] cordianted of orthocenter.

    if (np.sum(mask_infinite_vps) == 1):
        # Infiinite vanishing points assumed to be the third:
        principal_point = getPrincipalPoint2fntVP(vp)
    elif (np.sum(mask_infinite_vps) == 0):
        principal_point = getPrincipalPoint3fntVP(vp)
    else:
        print('ERROR: Two infinite vanishing points')

    return principal_point.reshape(1, 2)


def getPrincipalPoint2fntVP(vp):
    ref_point = vp[2, :]

    r = ((ref_point[0] - vp[0, 0]) * (vp[1, 0] - vp[0, 0]) \
         + (ref_point[1] - vp[0, 1]) * (vp[1, 1] - vp[0, 1])) / \
        ((vp[1, 0] - vp[0, 0]) ** 2 + (vp[1, 1] - vp[0, 1]) ** 2)

    principal_point = np.array([vp[0, 0] + r * (vp[1, 0] - vp[0, 0]),
                                vp[0, 1] + r * (vp[1, 1] - vp[0, 1])])
    return principal_point


def getPrincipalPoint3fntVP(vp):
    # Estiamtes prinicpal point assuming three infinite vanishing points.
    # The function estimates the thriangle orthocenter, where the triangle is
    # given by the three orthogonal vanishing points.
    # Finds the slope of two sides of the triangle m
    # Finds the slope of two perpenduclars as -1/m
    # Express line through opposite vp coordinates and perpendicular slope:
    # y - vp_y = (-1/m) * (x - vp_x);
    # Solve the system.

    # See proof that the prinipal point is an orthocenter:
    # "Roberto Cipolla, Tom Drummond, and Duncan P Robertson. 1999. Camera Calibration
    # from Vanishing Points in Image of Architectural Scenes.. In BMVC."

    side1 = vp[0, :] - vp[1, :]
    side2 = vp[0, :] - vp[2, :]

    A = np.vstack([side1, side2])

    b = np.array([np.dot(side1, vp[2, :]), np.dot(side2, vp[1, :])])

    principal_point = np.linalg.solve(A, b)
    return principal_point


def estimateFocalLengthFromVPsAndPrincipalPoint(vp, principal_point):
    # "Camera calibration using two or three vanishing points" by Orghidan et
    # al. Eq.(18)
    # Input:
    #   vp: 3x2 --- coordiantes of three vanishing points:
    principal_point = principal_point.reshape([1, 2])

    if vp.shape[1] > 3:
        vp = vp.reshape([2, 3])

    f = np.sqrt(np.abs(np.dot((vp[0, :] - principal_point).reshape(-1),
                              (principal_point - vp[1, :]).reshape(-1))))

    return f


def focalLength2FieldOfView(f, width):
    return 2.0 * np.degrees(np.arctan(width / (2.0 * f)))


def estimateRotationMatrix2VP(vp, principal_point, f):
    #  Eq (6) in the paper "Camera calibration using two or three vanishing points" by Orghidan et al.
    # principal_point = principal_point.reshape([1, 2])
    R = np.zeros([3, 3])
    R[:2, 0] = vp[0, :] - principal_point
    R[2, 0] = f
    R[:, 0] = R[:, 0] / np.linalg.norm(R[:, 0])
    # R[:,1] = [vp[1,:]-principal_point, f]
    R[:2, 1] = vp[1, :] - principal_point
    R[2, 1] = f
    R[:, 1] = R[:, 1] / np.linalg.norm(R[:, 1])
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    return R


def estimateRotationMatrix3VP(vp, f, principal_point):
    #   Recover scaling parameters of vanishing points in Eq. (14) using Eq. (22)
    #   Implements paper "Camera calibration using two or three vanishing points" by Orghidan et al.
    # 	Section III. Camera calibratino using three vanishing points.
    A = np.zeros([5, 3])
    A[0, :] = vp[:, 0]
    A[1, :] = vp[:, 1]
    A[2, :] = vp[:, 0] ** 2
    A[3, :] = vp[:, 1] ** 2
    A[4, :] = vp[:, 0] * vp[:, 1]

    b = np.zeros(5)
    b[:2] = principal_point
    b[2:4] = f ** 2 + principal_point ** 2
    # print(principal_point)
    b[4] = principal_point[0, 0] * principal_point[0, 1]
    # print("estimateRotationMatrix3VP")
    # print(A)
    # print(b)

    lambda_val = np.sqrt(np.linalg.pinv(A) @ b)
    # print("lambda_val")
    # print(lambda_val)

    R = np.zeros([3, 3])
    R[0, :] = lambda_val * (vp[:, 0] - principal_point[0, 0]) / f
    R[1, :] = lambda_val * (vp[:, 1] - principal_point[0, 1]) / f
    R[2, :] = lambda_val

    return R


def translationUpToScale(lambda_val, principal_point_vp,
                         point2DConterpartOrigin,
                         f):
    t = np.zeros(3)
    t[0] = lambda_val * (point2DConterpartOrigin[0] - principal_point_vp[0, 0]) / f
    t[1] = lambda_val * (point2DConterpartOrigin[1] - principal_point_vp[0, 1]) / f
    t[2] = lambda_val
    return t


def estimate_camera_parameters(vps, sketch_height, point_2d_counterpart_origin):
    cam_param = {}

    # Axis system:
    # +-1   0   0 -- first vp
    # 0   +-1   0 -- second vp
    # 0     0 +-1 -- third vp
    vp_, xyz_inds = findXYZ(vps)

    mask_infinite_vps = checkInfinityConditionVP(vp_, sketch_height)
    principal_point_vp = getPrincipalPoint(vp_, mask_infinite_vps)
    f_vp = estimateFocalLengthFromVPsAndPrincipalPoint(vp_, principal_point_vp)
    fov_vp = focalLength2FieldOfView(f_vp, sketch_height)

    # Rotation matrix:
    vp = vp_

    if sum(mask_infinite_vps) == 1:
        R_vp = estimateRotationMatrix2VP(vp, principal_point_vp, f_vp)
        # print('Two finite points')
        # print("estimateRotationMatrix2VP")
    elif sum(mask_infinite_vps) == 0:
        R_vp = estimateRotationMatrix3VP(vp, f_vp, principal_point_vp)
        # print('Three finite points')
    else:
        # print('Two vanishing points at infinity: near orthogonal projection')
        np.eye(3) * np.nan
        return None, None, None

    lambda_var = 1.0
    t_vp = translationUpToScale(lambda_var, principal_point_vp,
                                point_2d_counterpart_origin,
                                f_vp)
    t_vp = t_vp.reshape([3, 1])

    # Calibration matrix:
    K_vp = np.array([[f_vp, 0, principal_point_vp[0, 0], 0],
                     [0, f_vp, principal_point_vp[0, 1], 0],
                     [0, 0, 1.0, 0]])

    T_vp = np.zeros([4, 4])
    T_vp[:3, :3] = R_vp
    T_vp[:3, 3] = t_vp.reshape(3)
    T_vp[3, 3] = 1.0

    P_vp = K_vp @ T_vp

    cam_param["P"] = P_vp
    cam_param["f"] = f_vp
    cam_param["principal_point"] = principal_point_vp
    cam_param["R"] = R_vp
    cam_param["K"] = K_vp[:3, :3]
    cam_param["t"] = t_vp
    cam_param["view_dir"] = T_vp[2, :3]
    cam_param["fov"] = fov_vp
    cam_param["C"] = -np.linalg.pinv(R_vp) @ t_vp

    return cam_param, vp, np.array(xyz_inds, dtype=np.int64)


def estimate_initial_camera_parameters(vps, p, sketch):
    lines_group = []
    vp_new_ind = []
    ind_first_stroke = findFirstVisibleStrokeTowardsVP(p, sketch)
    point_2d_counterpart_origin = np.array(sketch.strokes[ind_first_stroke].primitive_geometry[[0, 2]])
    cam_param, vp, vp_new_ind = estimate_camera_parameters(vps, sketch.height, point_2d_counterpart_origin)
    cam_param["vp"] = vp_new_ind
    cam_param["vp_coord"] = vp
    p_ = p
    p_[:, :3] = p_[:, vp_new_ind]
    lines_group = np.argmax(p_, axis=1)

    return cam_param, lines_group, vp, vp_new_ind


def assignLineDirection(sketch, line_group):
    primitive_types = np.array([s.line_group for s in sketch.strokes])
    ind_lines = np.argwhere(primitive_types == 0).flatten()

    for s_id in range(len(sketch.strokes)):
        if sketch.strokes[s_id].is_curved:
            sketch.strokes[s_id].axis_label = 5
        else:
            sketch.strokes[s_id].axis_label = -1
    for j, i in enumerate(ind_lines):
        sketch.strokes[i].axis_label = line_group[j]
