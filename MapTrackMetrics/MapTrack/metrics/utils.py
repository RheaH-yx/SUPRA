import numpy as np

def calculate_wallPolygon(bounds, pWall=0.2):
    distTop = abs(bounds[0,0] - bounds[1,0])
    distBottom = abs(bounds[2,0] - bounds[3,0])
    distLeft = abs(bounds[0,1] - bounds[3,1])
    distRight = abs(bounds[1,1] - bounds[2,1])

    tl = int(bounds[0,0] + pWall*distTop), int(bounds[0,1] + pWall*distLeft)
    tr = int(bounds[1,0] - pWall*distTop), int(bounds[1,1] + pWall*distRight)
    br = int(bounds[2,0] - pWall*distBottom), int(bounds[2,1] - pWall*distRight)
    bl = int(bounds[3,0] + pWall*distBottom), int(bounds[3,1] - pWall*distLeft)
    points = np.array([tl,tr,br,bl])
    return points

def non_null_len(iterable):
    return len(list(filter(None, iterable)))
def arg_first_non_null(iterable):
    for i, el in enumerate(iterable):
        if el is not None:
            return i
def len_comparator(item1, item2):
    return non_null_len(item1) - non_null_len(item2)
def arg_first_comparator(item1, item2):
    return arg_first_non_null(item1) - arg_first_non_null(item2)

def get_odd(v):
    if v%2 ==0:
        return v-1
    return v