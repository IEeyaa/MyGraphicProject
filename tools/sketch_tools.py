import numpy as np
import svgpathtools
import svgwrite
from math import sqrt, atan, pi, tan
import itertools


class letter_generator:
	"""
	Iterative letter generator.

	"""
	def __init__(self):
		self.letter = 'a'

	def get(self):
		"""
		Get the next letter. Starts at A.

		:return: char
		"""
		l = self.letter
		self.letter = chr(ord(self.letter) + 1)
		return l


class number_generator:
	def __init__(self):
		self.number = 0

	def get(self):
		"""
		Get the next integer. Starts at 1.

		:return: int
		"""
		self.number += 1
		return self.number


def is_number(s):
	"""
	Check if s is a numeral.

    :param s: anything
    :return: true if s is a numeral
    """
	try:
		float(s)
		return True
	except ValueError:
		return False


def check_file_ext(file, ext):
	"""
    Check if filename has a specific extension

    :type file: string
    :type ext: string
    :return: true if file has extension ext
    """
	return (ext.lower() in file) or (ext.upper() in file)


def delete_consecutive_duplicates(L):
	"""
    yield a list ridden of its consecutive duplicates

    :type L: list
    :return: a list generator
    """
	prev = None
	for i, o in enumerate(L):
		if o != prev:
			prev = o
			yield i


def all_pairs_ind(L):
	"""
    returns all combinations of indices for list L

    :type L: list
    :return: a list of pairs of ints
    """
	return list(itertools.combinations(range(len(L)), 2))


def all_pairs(L):
	"""
    returns all pairs combinations for list L

    :type L: list
    :return: a list of pairs of objects in L
    """
	return [(L[id[0]], L[id[1]]) for id in all_pairs_ind(L)]


def unit_vector(a):
	"""
    normalizes a vector non destructively

    :param a: 2dvector
    :return: a / norm(a)
    """
	return a / np.linalg.norm(a)


def normal_point(p0, p1):
	"""
	Get a unit vector normal to a p0p1 segment.

	:param p0: (x,y)
	:param p1: (x,y)
	:return: (x,y)
	"""
	if all(p1.coords == p0.coords):
		return np.array([0, 0])
	v = p1.coords - p0.coords
	v /= np.linalg.norm(v)
	v = np.array([-v[1], v[0]])

	return v


def triangle_orthocenter(A, B, C):
	"""
    find the orthocenter of a triangle

    :param A: 2d point
    :param B: 2d point
    :param C: 2d point
    :return: 2d point, orthocenter of ABC
    """
	h1 = project_point_on_segment(B, C, A)
	h2 = project_point_on_segment(A, C, B)
	return get_intersection_2d(A, h1, B, h2)


def perp(a):
	"""
    create and return an orthogonal vector (no normalization)

    :param a: a = [ax, ay]
    :return: [-ay, ax]
    """
	b = np.empty_like(a)
	b[0] = -a[1]
	b[1] = a[0]
	return b


def get_intersection_2d(A, B, C, D):
	"""
    compute the intersection point of AB and CD analytically
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    :param A: 2d point
    :param B: 2d point
    :param C: 2d point
    :param D: 2d point
    :return: E in AB and in CD
    """
	da = B - A
	db = D - C
	dp = A - C
	dap = perp(da)
	denom = np.dot(dap, db)
	num = np.dot(dap, dp)
	return (num / denom.astype(float)) * db + C


def get_intersection_3d(P1, V1, P2, V2):
	"""
    compute the intersection point of 2 lines in 3D, assuming they are in a same plane
    # http://mathforum.org/library/drmath/view/62814.html

    :param P1: 3d point
    :param V1: 3d vector
    :param P2: 3d point
    :param V2: 3d vector
    :return: I = P1 + u1.V1 && I = P2 + v1.V2
    """
	V1 = V1 / np.linalg.norm(V1)
	V2 = V2 / np.linalg.norm(V2)
	cp1 = np.cross(V1, V2)
	if np.linalg.norm(cp1) == 0:
		print("error: lines do not intersect")
	cp2 = np.cross(P2 - P1, V2)
	a = np.linalg.norm(cp2) / np.linalg.norm(cp1)
	return P1 + a * V1


def project_point_on_segment(a, b, p, out_r=False):
	"""
    project point p on a segment [ab] orthogonally and return its coordinates I

    :param a: 2d point
    :param b: 2d point
    :param p: 2d point
    :param out_r: flag to also return the distance aI
    :return: I or (I, aI)
    """
	ap = p - a
	ab = b - a
	d = np.dot(ap, ab)
	lar = d / np.linalg.norm(ab)
	r = lar / np.linalg.norm(ab)
	result = a + r * ab
	if out_r:
		return result, r
	else:
		return result


def focal_length_to_fov(f, L):
	"""
    1 dimensionnal field of view formula

    :param f: focal length
    :param L: 2d distance from center in viewport coordinates
    :return: fov = 2 * arctan((L / 2) / f)
    """
	return 2 * np.arctan((L / 2) / f)


def project_point_3d(K, R, T, p):
	"""
    project a 3D point on a 2D viewport using calibrated camera

    :param K: 3x3 intrinsic matrix
    :param R: 3x3 rotation matrix
    :param T: 3x1 translation matrix
    :param p: 3x1 3d coordinates
    :return: P = K*[R|T] * p
    """
	P = np.zeros([4, 4])
	P[0:3, 0:3] = R
	P[0:3, 3] = T
	P[3, 3] = 1
	M = np.c_[K, [0, 0, 0]].dot(P)
	x = np.array([p[0], p[1], p[2], 1]).transpose()
	o = M.dot(x)
	o = o / o[2]
	return o[0:2]


def colinearity(u, v):
	"""
    colinearity factor between two vectors (normalized dotproduct)

    :param u: any 2d vector
    :param v: any 2d vector
    :return: (u/norm(u)).(v/norm(v))
    """
	uu = u / np.linalg.norm(u)
	vv = v / np.linalg.norm(v)
	return np.dot(uu, vv)


def normalize_list(a, mn=None, mx=None):
	"""
    normalize a list of scalars with either input or effective min and max

    :param a: list of scalars
    :param mn: optionnal minimum
    :param mx: optionnal maximum
    :return: normalized list
    """
	if mx == None: mx = np.max(a)
	if mn == None: mn = np.min(a)
	if mx - mn == 0:
		return [0] * len(a)
	return [(x - mn) / (mx - mn) for x in a]


def path_to_polylines(paths, polys):
	"""
    convert an svg path to a polyline using svgpathtools

    :param paths: list of paths
    :param polys: out argument : the new polylines
    """
	paths, attributes, svg_attributes = svgpathtools.svg2paths(paths, return_svg_attributes=True)
	dwg = svgwrite.Drawing(size=(svg_attributes["width"], svg_attributes["height"]))

	for path in paths:
		polyline = dwg.polyline(stroke="black", stroke_width=1.0)
		for p in path:
			if isinstance(p, svgpathtools.Line):
				polyline.points.append((p[0].real, p[0].imag))
				polyline.points.append((p[1].real, p[1].imag))
			else:
				# other classes: Arc, QuadraticBezier, CubicBezier
				print("Not implemented yet!")
		dwg.add(polyline)

	dwg.saveas(polys, pretty=True, indent=True)
