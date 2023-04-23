#! python
import math

# sources of inspiration: https://youtu.be/DlPrTGbO19E,
# https://github.com/Vectorized/Python-KD-Tree/blob/master/kd_tree.py


class KDtree:

    def __init__(self, points_coor: list[list[float]], dim: int):
        self.dim = dim
        self._check_list(points_coor)
        self.points, self.root = points_coor.copy(), None
        self._build_tree(self.points)

    def _build_tree(self, points_coor: list[list[float]], depth=0):

        if len(points_coor) > 2:
            axis = depth % self.dim
            brk = len(points_coor) // 2
            points_coor.sort(key=lambda x: x[axis])
            self._add_node(points_coor[brk])
            self._build_tree(points_coor[:brk], depth + 1)
            self._build_tree(points_coor[brk + 1:], depth + 1)
        else:
            for f in points_coor:
                self._add_node(f)

    def _check_list(self, points_coor: list[list[float]]):
        for f in points_coor:
            if len(f) != self.dim:
                raise ValueError('every points should have ', self.dim, ' coordinates')

    def _add_node(self, coor: list):
        if len(coor) != self.dim:
            raise ValueError('point must have ', self.dim, ' coordinates')
        point_to_add = NodeKD(coor, self)  # делаем нод тут, чтобы если пустой рут
        if not self.root:
            self.root = point_to_add
        else:
            point_to_add._fit_in(self.root)

    def find_nearest(self, point: list):
        return _find_nearest(point, self.root, self.dim)

    def find_neighbors(self, point: list, radius: float) -> dict:
        return _find_neighb(point, self.root, self.dim, radius)


class NodeKD:
    # nodes for kdtree
    # self.level can be 0 for x and 1 for y
    def __init__(self, coor: list, kdtree: KDtree):
        self.parent, self.right, self.left = None, None, None
        self.coor, self.kdtree = coor, kdtree
        self.level = None

    def _fit_in(self, node, depth=0):  # тут не могу сослаться на этот же класс
        if type(node) != NodeKD:
            raise ValueError('starting node must be NodeKD class')
        axis = depth % self.kdtree.dim
        if self.coor[axis] >= node.coor[axis]:
            if node.right:
                self._fit_in(node.right, depth + 1)
            else:
                node.right = self
                self.parent = node
        else:
            if node.left:
                self._fit_in(node.left, depth + 1)
            else:
                node.left = self
                self.parent = node


def closets_point(target_p: list, p1: NodeKD | None, p2: NodeKD | None) -> NodeKD | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    d1 = distance(target_p, p1.coor)
    d2 = distance(target_p, p2.coor)
    if d1 < d2:
        return p1
    else:
        return p2


# calculate euclidean distance
def distance(a: list, b: list):
    dist = 0
    for i, f in enumerate(a):
        dist += (f - b[i]) ** 2
    return math.sqrt(dist)


def _find_nearest(point: list, node: NodeKD | None, dim, depth=0):
    if node is None:
        return None
    axis = depth % dim
    next_branch = None
    opposite_branch = None
    if point[axis] >= node.coor[axis]:
        next_branch = node.right
        opposite_branch = node.left
    else:
        next_branch = node.left
        opposite_branch = node.right
    best = closets_point(point, _find_nearest(point, next_branch, dim, depth + 1), node)
    # прошли по всей ветке, нашли лучшую в ней
    if distance(point, best.coor) > abs(point[axis] - node.coor[axis]):
        best = closets_point(point, _find_nearest(point, opposite_branch, dim, depth + 1), best)
        # идём по другой ветке и сравниваем
    return best


def _find_neighb(point: list, node: NodeKD | None, dim: int, radius: float, depth=0):
    if node is None:
        return None
    axis = depth % dim

    next_branch = None
    opposite_branch = None
    if point[axis] >= node.coor[axis]:
        next_branch = node.right  # сюда точно идём
        opposite_branch = node.left  # сюда возможно идём
    else:
        next_branch = node.left
        opposite_branch = node.right

    neighb_points = {}
    current_dist = distance(point, node.coor)
    if current_dist < radius:
        neighb_points[node] = current_dist
    take_other = _find_neighb(point, next_branch, dim, radius, depth + 1)  # прошли по всей ветке, нашли все
    if take_other:
        neighb_points.update(take_other)
    if abs(point[axis] - node.coor[axis]) < radius:
        take_other = _find_neighb(point, opposite_branch, dim, radius, depth + 1)  # идём по другой ветке и ищем все
        if take_other:
            neighb_points.update(take_other)
    return neighb_points
