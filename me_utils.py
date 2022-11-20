import math


def compute_delta(i1, j1, i2, j2):
    """
    compute line angle delta according to 2 line points (i1,j1), (i2,j2)
    the result value lies between -pi and pi
    """
    dif_i = i2 - i1
    dif_j = j2 - j1
    delta = math.atan2(dif_j, dif_i)
    while delta > math.pi:
        delta -= math.pi
    while delta < -math.pi:
        delta += math.pi
    return delta


def is_out_of_img(i, j, half_mask, width, height):
    # 抄的visp, 2可能是为了边缘不要抵到头
    return not ((i - half_mask > 2) and (i + half_mask < width-2)
                and (j - half_mask > 2) and (j + half_mask < height-2))