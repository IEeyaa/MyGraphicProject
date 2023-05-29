import numpy as np
from copy import deepcopy
from pylowstroke.sketch_trimming import trimLineCoordinatesDrawingField
from pylowstroke.sketch_camera import assignLineDirection

def get_line_fitting(x, y):
    ind = np.argmax([np.max(x)-np.min(x), np.max(y)-np.min(y)])
    if (ind == 0):
        f = np.polyfit(x, y, deg=1)
    else:
        f = np.polyfit(y, x, deg=1)

    if (ind == 0):
        x1 = np.min(x)
        ind_x1 = np.argmin(x)
        
        x2 = np.max(x)
        ind_x2 = np.argmax(x)
        y1 = np.poly1d(f)(x1)
        y2 = np.poly1d(f)(x2)

        # Ensure that the line endpoints are in the same order as in the
        # polyline:
        if (ind_x1 < ind_x2):
            line = [x1, x2, y1, y2,]
        else:
            line = [x2, x1, y2, y1]
    else:
        y1 = np.min(y)
        ind_y1 = np.argmin(y)
        y2 = np.max(y)
        ind_y2 = np.argmax(y)
        x1 = np.poly1d(f)(y1)
        x2 = np.poly1d(f)(y2)

        # Ensure that the line endpoints are in the same order as in the
        # polyline:
        if (ind_y1 < ind_y2):
            line = [x1, x2, y1, y2]
        else:
            line = [x2, x1, y2, y1]

    return np.array(line)

def compute_intersection_points(lines):
    # End points of line segments:
    # The points are assumed to lie on the plane that haz z = 1
    p1 = np.ones([len(lines), 3])
    p2 = np.ones([len(lines), 3])
    p1[:, :2] = lines[:, [0, 2]]
    p2[:, :2] = lines[:, [1, 3]]
    # Get normals to the planes defined by the line segments 
    l = np.cross(p1, p2)
    l = l / np.expand_dims(np.sqrt(np.sum(l**2, axis=-1)), 1)

    # Get the vector that is common for both planes and line in the image plane
    XX, YY = np.meshgrid(range(len(l)), range(len(l)))
    ll1 = l[XX[:],:].reshape(-1, 3)
    ll2 = l[YY[:],:].reshape(-1, 3)
    Xpnts = np.cross(ll1,ll2)

    #[x1 y1 x2 y2 x3 y3] are colinear if x1(y2-y3)+x2(y3-y1)+x3(y1-y2)=0;
    colchck = np.zeros([Xpnts.shape[0], 6])
    colchck[:, 0] = lines[XX.transpose().reshape(-1),0]
    colchck[:, 1] = lines[XX.transpose().reshape(-1),2]
    colchck[:, 2] = lines[YY.transpose().reshape(-1),0]
    colchck[:, 3] = lines[YY.transpose().reshape(-1),2]
    colchck[:, 4] = lines[YY.transpose().reshape(-1),1]
    colchck[:, 5] = lines[YY.transpose().reshape(-1),3]
    colchck = colchck[:,0]*(colchck[:,3]-colchck[:,5])+colchck[:,2]*(colchck[:,5]-colchck[:,1]) + \
        colchck[:,4]*(colchck[:,1]-colchck[:,3])

    keepind = np.argwhere(np.abs(colchck)>50).flatten()
    Xpnts = Xpnts[keepind, :]
    new_Xpnts = np.zeros([len(Xpnts),2])
    new_Xpnts[:,0] = Xpnts[:,0]/Xpnts[:,2]
    new_Xpnts[:,1] = Xpnts[:,1]/Xpnts[:,2]
    return new_Xpnts

def computeLinesPointsVotes(lines, Xpnts):

    ta = np.pi/3 #threshold on line to belong to the  pi/6 failure cases

    distances = np.linalg.norm(lines[:, [0,2]]-lines[:,[1,3]], axis=-1)
    max_length = np.max(distances)

    VoteArr=np.zeros([len(lines),len(Xpnts)])
    for line_num in range(len(lines)):
        theta, vp_outside_line_segment = computeDistanceLineVPs(lines[line_num],Xpnts)
        indd = np.argwhere(np.logical_and(theta < ta, vp_outside_line_segment)).flatten()
        VoteArr[line_num,indd] = (1-theta[indd]/ta)

    VoteArr = np.exp(-(1-VoteArr)**2/0.1/0.1/2)
    VoteArr = np.multiply(VoteArr, np.repeat(distances.reshape(-1, 1), VoteArr.shape[1], axis=1)/max_length)
    return VoteArr

def computeDistanceLineVPs(segment_s, vp):

    x1=segment_s[0]
    x2=segment_s[1]
    y1=segment_s[2]
    y2=segment_s[3]

    midpnt = np.array([(x1+x2)/2, (y1+y2)/2])
    lengthl = np.sqrt((x1-x2)**2+(y1-y2)**2)

    slope1 = (y2-y1)/(x2-x1)

    # Line segment s_dash (Fig 3 a)
    slope2 = (vp[:,1]-np.ones(len(vp))*midpnt[1])/(vp[:,0]-np.ones(len(vp))*midpnt[0])

    # Angle between s and s_dash
    slope1 = np.ones(len(slope2))*slope1
    theta=np.arctan(np.abs((slope1-slope2)/(1+slope1*slope2)))

    #Discard vanishing points that lie withing the endpoints of s' (segment_s
    #rotated to lie on line towards vanishing point)
    # Distance between midpoint and vanishing point:
    d = np.sqrt((vp[:,0]-np.ones(len(vp))*midpnt[0])**2+(vp[:,1]-np.ones(len(vp))*midpnt[1])**2)
    vp_outside_line_segment = np.zeros(len(vp), dtype=np.bool)
    vp_outside_line_segment[:] = False
    vp_outside_line_segment[d >= 0.95*lengthl/2] = True

    return theta, vp_outside_line_segment

#def computeVPGivenParallelLines(lines):
#    num_lines = len(lines)
#   
#    x1 = lines[:, 0]
#    x2 = lines[:, 1]
#    y1 = lines[:, 2]
#    y2 = lines[:, 3]
#    
#    b = (x2 - x1)*y1 + (y1 - y2)*x1
#    A = np.zeros([num_lines, 2])
#    A[:,0] = (y1-y2)
#    A[:,1] = (x2-x1)
#    
#    vp = A[:, 0]/b
#    return vp

def removeRedundantPoints(Xpnts,w,h):
    # Remove points that are too close to image center
    #print(len(Xpnts))
    inds = np.argwhere(
        np.logical_or(Xpnts[:,0] > 0.8*w ,
                    np.logical_or(Xpnts[:,0] < 0.2*w,
                    np.logical_or(
           Xpnts[:,1] > 0.8*h, Xpnts[:,1] < 0.2*h)))).flatten()
    Xpnts = Xpnts[inds]

    # Remove redundant points
    currid = 0

    Xpnts2 = deepcopy(Xpnts)

    while len(Xpnts) > 0:
        Xpnts2[currid, :] = Xpnts[0,:]
        dists = (Xpnts[0,0]-Xpnts[:,0])**2 + (Xpnts[0,1]-Xpnts[:,1])**2

        center_dist = np.sqrt((Xpnts[0,0]-w)**2+(Xpnts[0,1]-h)**2)/np.sqrt((w/2)**2+(h/2)**2)
        if center_dist < 1.0:
            # Higher precision close to the sketch:
            thres = 20
        else:
            # Precision thta varies as we go further from the center:
            thres = 30*center_dist

        inds = np.argwhere(dists < thres).flatten()
        num_points = len(inds)
        Xpnts2[currid,0] = np.sum(Xpnts[inds,0])/num_points
        Xpnts2[currid,1] = np.sum(Xpnts[inds,1])/num_points

        inds = np.argwhere(dists>thres).flatten()
        Xpnts = Xpnts[inds]

        currid += 1

    Xpnts2 = Xpnts2[:currid]
    return Xpnts2

def check_orthogonality(vp1s,vp2s,vp3s,w,h):
    # vp1s, vp2s, and vp3s multiple vanishing points.

    orthocheck = np.zeros(len(vp1s))

    #inf conditions
    inf1 = np.logical_or(vp1s[:,0]>10*w, vp1s[:,1]>10*h)
    inf2 = np.logical_or(vp2s[:,0]>10*w, vp2s[:,1]>10*h)
    inf3 = np.logical_or(vp3s[:,0]>10*w, vp3s[:,1]>10*h)
    inds_fff = np.argwhere(
        np.logical_and(np.logical_not(inf1), 
            np.logical_and(np.logical_not(inf2), np.logical_not(inf3)))).flatten()
    # Rearrange to have third point at in infinity
    inds = np.argwhere(
        np.logical_and(inf1, np.logical_and(np.logical_not(inf2), np.logical_not(inf3)))).flatten()
    temp = vp1s[inds]
    vp1s[inds,:] = vp3s[inds,:]
    vp3s[inds,:] = temp
    inds = np.argwhere(
        np.logical_and(np.logical_not(inf1), np.logical_and(inf2, np.logical_not(inf3)))).flatten()
    temp = vp2s[inds,:]
    vp2s[inds,:] = vp3s[inds,:]
    vp3s[inds,:] = temp

    # We moved infinite point to the third row
    inds_ffi = np.argwhere((inf1+inf2+inf3)==1).flatten() # Only one point is at infinity

    inds = np.argwhere(
        np.logical_and(inf1, np.logical_and(np.logical_not(inf2), np.logical_not(inf3)))).flatten()

    # Rearrange to make 2nd and 3rd point being at infinity
    temp = vp2s[inds]
    vp2s[inds,:] = vp1s[inds,:]
    vp1s[inds,:] = temp
    inds = np.argwhere(
        np.logical_and(inf1, np.logical_and(inf2, np.logical_not(inf3)))).flatten()
    temp = vp3s[inds,:]
    vp3s[inds,:] = vp1s[inds,:]
    vp1s[inds,:] = temp
    inds_fii = np.argwhere((np.array(inf1,dtype=np.int)+np.array(inf2,dtype=np.int)+np.array(inf3, dtype=np.int))==2).flatten()
    inds_iii = np.argwhere(np.logical_and(inf1, np.logical_and(inf2, inf3))).flatten()

    if np.prod(inds_fff.shape)>0:
        #print("3finitvp")
        fff_orthochk = orthochk_3finite_vp(vp1s,vp2s,vp3s,inds_fff,w,h)
        orthocheck[inds_fff] = fff_orthochk

    if np.prod(inds_ffi.shape)>0:
        #print("2finitvp")
        ffi_orthochk = orthochk_2finite_vp(vp1s,vp2s,vp3s,inds_ffi,w,h)
        orthocheck[inds_ffi] = ffi_orthochk

    if np.prod(inds_fii.shape)>0:
        #print("1finitvp")
        v1sf = vp1s[inds_fii,:]
        v2si = vp2s[inds_fii,:]
        v3si = vp3s[inds_fii,:]

        temp_orthochk = np.zeros(inds_fii.shape)

        vec1 = np.vstack([v2si[:,0], v2si[:,1]])
        vec2 = np.vstack([v3si[:,0], v3si[:,1]])

        dot12 = np.sum(np.einsum('ij,ij->ij', vec1, vec2), axis=0)
        norm1 = np.sum(np.einsum('ij,ij->ij', vec1, vec1), axis=0)
        norm2 = np.sum(np.einsum('ij,ij->ij', vec2, vec2), axis=0)
    
        inds = np.logical_and(np.abs(dot12/norm1/norm2)<=0.1,
            ind_feasible_principle_point_mild(v1sf[:,0], v1sf[:,1], w, h))
        temp_orthochk[inds] = True

        orthocheck[inds_fii] = temp_orthochk

    if np.prod(inds_iii.shape)>0:
        orthocheck[inds_iii] = 0
    
    return orthocheck

def orthochk_2finite_vp(vp1s,vp2s,vp3s,inds_ffi,w,h):
    ffi_orthochk = np.zeros(inds_ffi.shape, dtype=np.bool)

    v1sf = vp1s[inds_ffi,:]
    v2sf = vp2s[inds_ffi,:]
    v3si = vp3s[inds_ffi,:]

    # I. Find principal point and check its feasibility:
    # In the original Rother's paper the point is selected lying on the
    # line between the two finite vanishing points and closer to the image
    # center. 
    # In our case it makes sense to allign the center with the cneter mass
    # of the sketched shape itself.
      
    ref_point = v3si
    
    # Find projection of the infinite vanishing point on the line
    # connecting the two vanishing points:
    r = ((ref_point[:,0]-v1sf[:,0])*(v2sf[:,0]-v1sf[:,0])+\
        (ref_point[:,1]-v1sf[:,1])*(v2sf[:,1]-v1sf[:,1]))/((v2sf[:,0]-v1sf[:,0])**2+(v2sf[:,1]-v1sf[:,1])**2)    
    
    u0= v1sf[:,0] + r*(v2sf[:,0]-v1sf[:,0])
    v0= v1sf[:,1] + r*(v2sf[:,1]-v1sf[:,1])
    
    point_is_between_two_finite_points = np.logical_and(r > 0, r < 1)
    ind_feasible_principal_point = np.logical_and(point_is_between_two_finite_points,
        ind_feasible_principle_point_2finitevp(u0, w))
    
    principal_point = np.zeros([len(inds_ffi), 2])
    principal_point[:,0] = u0
    principal_point[:,1] = v0
    
    # II. Find focal length and check its feasibility:
    vec1 = v1sf - principal_point
    vec2 = principal_point - v2sf    
    f = np.sqrt(np.abs(vector_dot(vec1,vec2))) 
    fov = 2*np.degrees(np.arctan((w/(2.0*f))))
    inf_feasible_fov = np.logical_and(fov > 10, fov < 120)
    
    inds = np.logical_and(ind_feasible_principal_point, inf_feasible_fov)

    ffi_orthochk[inds] = True
    return ffi_orthochk


def orthochk_3finite_vp(vp1s,vp2s,vp3s,inds_fff,w,h):
    fff_orthochk = np.zeros(inds_fff.shape, dtype=np.bool)    
    fff_orthochk[:] = False
    
    v1sf = vp1s[inds_fff,:]
    v2sf = vp2s[inds_fff,:]
    v3sf = vp3s[inds_fff,:]
    
    #I. Check orthogonality condition:
    inds_orthogonal = np.argwhere(checkOrthogonalityCriterion(v1sf, v2sf, v3sf)).flatten()
    v1sf = v1sf[inds_orthogonal, :]
    v2sf = v2sf[inds_orthogonal, :]
    v3sf = v3sf[inds_orthogonal, :]
        
    #Check the rest of conditions only on those points that passed intial
    #test on orthogonality:
    
    #II. Find principal point:
    principal_point = findPrincipalPointFocalLength3VP(v1sf, v2sf, v3sf)
    u0 = principal_point[:,0]
    v0 = principal_point[:,1]
    ind_feasible_principal_point = ind_feasible_principle_point(u0, v0, w, h)
    
    #III. Find focal length and check that it is feasible:
    vec1 = v1sf - principal_point
    vec2 = principal_point - v2sf
    f = np.sqrt(np.abs(vector_dot(vec1,vec2)))
    fov = 2*np.degrees(np.arctan((w/(2.0*f))))
    inf_feasible_fov = np.logical_and(fov > 10, fov < 120)
    
    inds = np.logical_and(ind_feasible_principal_point, inf_feasible_fov)
    inds = inds_orthogonal[inds]
    fff_orthochk[inds] = True
    return fff_orthochk

def ind_feasible_principle_point(u0, v0, w, h):
    ind = np.logical_and(u0 <= 0.7*w, 
        np.logical_and(u0 >= 0.3*w,
            np.logical_and(v0 <= 0.7*h, v0 >= 0.3*h)))
    return ind

def ind_feasible_principle_point_2finitevp(u0, w):
    #For now only width is considered since in case of vertical infinite
    #vanishing point the h can be out of the omage plane for the principle
    #point.
    ind = np.logical_and(u0 <= 0.7*w, u0 >= 0.3*w)
    return ind

def ind_feasible_principle_point_mild(u0, v0, w, h):
    ind = np.logical_and(u0 <= 0.7*w,
        np.logical_and(u0 >= 0.3*w,
            np.logical_and(v0 <= 0.7*h, v0 >= 0.3*h)))
    return ind


def findPrincipalPointFocalLength3VP(v1sf, v2sf, v3sf):
    #Find principal point given triplets of three finite vanishing points.
    Mats_11 = v1sf[:,0]+v2sf[:,0]
    Mats_12 = v1sf[:,1]+v2sf[:,1]
    Mats_13 = v1sf[:,0]*v2sf[:,0]+v1sf[:,1]*v2sf[:,1]
    Mats_21 = v1sf[:,0]+v3sf[:,0]
    Mats_22 = v1sf[:,1]+v3sf[:,1]
    Mats_23 = v1sf[:,0]*v3sf[:,0]+v1sf[:,1]*v3sf[:,1]
    Mats_31 = v3sf[:,0]+v2sf[:,0]
    Mats_32 = v3sf[:,1]+v2sf[:,1]
    Mats_33 = v3sf[:,0]*v2sf[:,0]+v3sf[:,1]*v2sf[:,1]

    A_11 = Mats_11-Mats_21
    A_12 = Mats_12-Mats_22
    A_21 = Mats_11-Mats_31
    A_22 = Mats_12-Mats_32
    b_1 = Mats_13-Mats_23
    b_2 = Mats_13-Mats_33
    detA = A_11*A_22-A_12*A_21
    principal_point = np.zeros([len(v1sf), 2])
    principal_point[:,0] = [A_22*b_1-A_12*b_2]/detA
    principal_point[:,1] = [A_11*b_2-A_21*b_1]/detA

    return principal_point


def vector_dot(a, b):
    return np.sum(np.einsum('ij,ij->ij', a, b), axis=1)

def checkOrthogonalityCriterion(v1sf, v2sf, v3sf):
    side1 = (v1sf - v2sf)
    side2 = (v1sf - v3sf)
    side3 = (v2sf - v3sf)
    
    side1 = side1/np.linalg.norm(side1, axis=1).reshape(-1, 1)
    side2 = side2/np.linalg.norm(side2, axis=1).reshape(-1, 1)
    side3 = side3/np.linalg.norm(side3, axis=1).reshape(-1, 1)
    
    theta1 = np.arccos(vector_dot(side1,side2))
    theta2 = np.arccos(vector_dot(-side3,-side2))
    theta3 = np.arccos(vector_dot(-side1,side3))

    inds = np.logical_and(theta1 < np.pi/2,
        np.logical_and(theta2 < np.pi/2, theta3 < np.pi/2))
    return inds

def get_line_primitives(sketch):
    all_lines = []
    for s_id, s in enumerate(sketch.strokes):

        if not s.is_curved():

            line_fitting = get_line_fitting(
                np.array([p[0] for p in s.lineString]),
                np.array([p[1] for p in s.lineString]))

            x_vals, y_vals, inside = trimLineCoordinatesDrawingField(
                line_fitting[:2], line_fitting[2:], sketch.height)

            all_lines.append(np.hstack([x_vals, y_vals]))
            s.primitive_geometry = line_fitting
            s.line_group = 0
        else:
            s.line_group = 1
    # compute line intersections
    all_lines = np.array(all_lines)
    return all_lines

def get_vanishing_points(sketch):
    failed = False
    all_lines = get_line_primitives(sketch)
    #print(len(sketch.strokes))
    #print("all_lines")
    #print(all_lines[:])
    #for l in all_lines:
    #    print(l)
    #exit()
    #print("len(all_lines)")
    #print(len(all_lines))
    x_pts = compute_intersection_points(all_lines)
    #print("len(x_pts)")
    #print(len(x_pts))
    vote_arr = computeLinesPointsVotes(all_lines, x_pts)
    vote = np.sum(vote_arr, axis=0)
    sorted_indices = np.flip(np.argsort(vote))
    vp_selected = x_pts[sorted_indices[0]]

    lines_votes_vp1 = vote_arr[:, sorted_indices[0]]
    # remove scaling
    distances = np.linalg.norm(all_lines[:, [0,2]]-all_lines[:,[1,3]], axis=-1)
    max_length = np.max(distances)
    active_lines = np.argwhere(lines_votes_vp1*(max_length/distances) < 0.8).flatten()
    inactive_lines = np.argwhere(lines_votes_vp1*(max_length/distances) >= 0.8).flatten()
    lines_votes_vp1 = np.array(list(lines_votes_vp1[active_lines]) + list(lines_votes_vp1[inactive_lines]))
    #vp = computeVPGivenParallelLines(all_lines[inactive_lines])
    #print(vp)
    vp = vp_selected

    # work with the remaining lines
    lines = all_lines[active_lines]
    #print(len(lines))
    x_pts = compute_intersection_points(lines)
    x_pts = removeRedundantPoints(x_pts, sketch.width, sketch.height)
    #print(x_pts)
    #print(len(x_pts))
    vote_arr = computeLinesPointsVotes(np.vstack([lines, all_lines[inactive_lines]]), x_pts)
    vote = np.sum(vote_arr[:len(lines)], axis=0)
    vv = np.flip(np.sort(vote))
    sorted_indices = np.flip(np.argsort(vote))
    vote = vv
    x_pts = x_pts[sorted_indices]
    vote_arr = vote_arr[:, sorted_indices]

    tri_mat = np.triu(np.ones([len(vote), len(vote)]), k=1)
    pts_ids = np.argwhere(tri_mat>0)
    pts_2 = pts_ids[:, 1]
    pts_1 = pts_ids[:, 0]
    npts = len(pts_1)
    tmp_step = 100000
    orthochecks = []
    for pt in np.arange(0, npts, step=tmp_step):
        temp_inds = np.arange(pt, np.minimum(pt+tmp_step, npts))
        tmp_vps = np.ones([len(temp_inds), 2])
        tmp_vps[:] = vp
        temp_orthochecks = check_orthogonality(
            tmp_vps,
            x_pts[pts_1[temp_inds]],
            x_pts[pts_2[temp_inds]],
            sketch.width, sketch.height
        )
        orthochecks += list(temp_orthochecks)

    orthos = np.argwhere(orthochecks).flatten()
    pts_1 = pts_1[orthos]
    pts_2 = pts_2[orthos]
    npts = len(pts_1)

    # Total vote computation for these points
    totVote = np.zeros(npts)
    for ln in range(len(lines_votes_vp1)):
        Votes = np.hstack([lines_votes_vp1[ln]*np.ones(npts).reshape(-1,1), vote_arr[ln,pts_1].reshape(-1,1), vote_arr[ln,pts_2].reshape(-1,1)])
        Votes = np.max(Votes, axis=1) # Only one line votes for the vanishing point, the one with maximum score
        totVote = totVote+Votes
    totVote = np.hstack([pts_1.reshape(-1,1), pts_2.reshape(-1,1), totVote.reshape(-1,1)])

    if len(totVote) > 0:
        ii = np.flip(np.argsort(totVote[:, 2]))
        vp_2 = x_pts[int(totVote[ii[0], 0])]
        vp_3 = x_pts[int(totVote[ii[0], 1])]

        VoteArrTemp = computeLinesPointsVotes(all_lines, np.array([vp, vp_2, vp_3]))
        p = np.hstack([VoteArrTemp*max_length/np.repeat(distances.reshape(-1,1), 3, axis=1),
            np.zeros(len(distances)).reshape(-1,1)])
        ind = np.argwhere(np.max(p[:,:3],axis=1)< 0.5).flatten()
        p[ind,3] = 1
        p /= np.sum(p,axis=1).reshape(-1,1)
        vps = np.array([vp, vp_2, vp_3])
    else:
        failed = True
        vps = np.array([])
        p = []

    return vps, p, failed

def assign_line_directions_given_vps(sketch, vps):
    all_lines = np.array(get_line_primitives(sketch))
    VoteArrTemp = computeLinesPointsVotes(all_lines, np.array(vps))
    distances = np.linalg.norm(all_lines[:, [0,2]]-all_lines[:,[1,3]], axis=-1)
    max_length = np.max(distances)
    p = np.hstack([VoteArrTemp*max_length/np.repeat(distances.reshape(-1,1), 3, axis=1),
        np.zeros(len(distances)).reshape(-1,1)]); #4th vp is outliers
    ind = np.argwhere(np.max(p[:,:3],axis=1)< 0.5).flatten()
    p[ind,3] = 1
    p /= np.sum(p,axis=1).reshape(-1,1)
    #print(p)
    lines_group = np.argmax(p, axis=1)
    assignLineDirection(sketch, lines_group)
    #for s_i, s in enumerate(sketch.strokes):
    #    print(s_i, s.axis_label)