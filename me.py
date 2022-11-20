import numpy as np
import math


def line_trough_pixel(P, Q, min_x, min_y, max_x, max_y):
    line_y_diff = Q[1] - P[1]
    line_x_diff = Q[0] - P[0]

    # judge if the four corner is at the same side of P,Q
    conner1 = min_y*line_x_diff - min_x*line_y_diff
    conner2 = min_y*line_x_diff - max_x*line_y_diff
    conner3 = max_y*line_x_diff - min_x*line_y_diff
    conner4 = max_y*line_x_diff - max_x*line_y_diff

    if (conner1 >= 0 and conner2 >= 0 and conner3 >= 0 and conner4 >= 0) or \
       (conner1 <= 0 and conner2 <= 0 and conner3 <= 0 and conner4 <= 0):
        return False, None, None
    else:
        if line_x_diff == 0:
            # vertical line across a single pixel
            new_P = (P[0], min_y)
            new_Q = (Q[0], max_y)
            return True, new_P, new_Q
        if line_y_diff == 0:
            # vertical line across a single pixel
            new_P = (min_x, P[1])
            new_Q = (max_x, Q[1])
            return True, new_P, new_Q
        # compute intersection of PQ and x=minx/y=miny/x=maxx/y=maxy separately
        inter1 = (min_x, min_x*line_y_diff/line_x_diff)
        inter2 = (max_x, max_x*line_y_diff/line_x_diff)
        inter3 = (min_y*line_x_diff/line_y_diff, min_y)
        inter4 = (max_y*line_x_diff/line_y_diff, max_y)
        intersects = [inter1, inter2, inter3, inter4]
        intersects.sort(key=(lambda point: point[0]))
        # the 2 points in the middle of 4 intersection points must be the clipped P, Q
        new_P, new_Q = intersects[1], intersects[2]
        return True, new_P, new_Q


def compute_pix_value(P, Q, min_x, min_y, max_x, max_y):
    # the pixel value is the area difference between 2 segments of the pixel
    if P[0] > Q[0]:
        tmp = P
        P = Q
        Q = tmp

    if P[0] == min_x and Q[0] == max_x:
        return math.fabs(max_y + min_y - P[1] - Q[1])
    elif (P[1] == min_y and Q[1] == max_y) or (Q[1] == min_y and P[1] == max_y):
        return math.fabs(max_x + min_x - P[0] - Q[0])
    elif P[0] == min_x and Q[1] == max_y:
        return 1-(max_y - P[1])*(Q[0] - min_x)
    elif P[0] == min_x and Q[1] == min_y:
        return 1-(P[1] - min_y)*(Q[0] - min_x)
    elif P[1] == min_y and Q[0] == max_x:
        return 1-(max_x-P[0])*(Q[1]-min_y)
    elif P[1] == max_y and Q[0] == max_x:
        return 1-(max_x-P[0])*(max_y-Q[1])
    else:
        raise Exception(f"\nP: {P},Q: {Q} is not clipped on the pixel boundaries: \
                        x={min_x}, y={min_y}, x={max_x}, y={max_y}\n")


def calcul_masks(angles, mask_size, masks):
    half_mask = float(mask_size)/2.
    n_theta = len(angles)
    for i_theta in range(n_theta):
        theta = math.radians(angles[i_theta])
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # define a line segment PQ crossing the mask center with angle theta
        # all the points are based on coordinate center at mask center
        if angles[i_theta] == 90:
            # tan(theta) is infinite
            P = (0., -float(mask_size))
            Q = (0., float(mask_size))
        else:
            tan_theta = sin_theta / cos_theta
            P = (-float(mask_size), tan_theta * (-mask_size))
            Q = (float(mask_size), tan_theta * mask_size)

        # judge the sign and value of every pixel, start from left-top pixel

        cur_pix_y = -half_mask + 0.5
        for i in range(mask_size):
            cur_pix_x = -half_mask + 0.5
            for j in range(mask_size):
                # judge the sign of pixels
                sign = np.sign(cos_theta * cur_pix_y - sin_theta * cur_pix_x)

                # judge if the line segment PQ goes through the pixel
                is_through_pixel, clip_P, clip_Q = line_trough_pixel(P, Q,
                                                                     cur_pix_x-0.5, cur_pix_y-0.5,
                                                                     cur_pix_x+0.5, cur_pix_y+0.5)
                if is_through_pixel:
                    # pixel value is the area difference between the 2 segments divided by PQ
                    value = compute_pix_value(clip_P, clip_Q,
                                              cur_pix_x-0.5, cur_pix_y-0.5,
                                              cur_pix_x+0.5, cur_pix_y+0.5)
                else:
                    value = 1.

                # for practical reason scale value 100 times and round to integer
                masks[i_theta][i, j] = round(100 * sign * value)
                cur_pix_x += 1.
            cur_pix_y += 1.


class Me:
    def __init__(self):
        # hyper paras
        self.range = 5
        self.mu1 = 0.5
        self.mu2 = 0.5
        self.threshold = 2500
        self.angle_step = 1
        self.mask_size = 3
        self.strip = 2
        self.mask = []
        self.init_mask()


    def init_mask(self):
        """
        init convolution mask for a specif line angle,
        imagine the line goes through the mask center,
        it divides the mask into value 1 areas and value -1 area,
        the tricky part is for those pixels that the line gos thorough,
        its value is the area difference between the split parts.
        """
        n_mask = int(180/self.angle_step)
        if self.mask:
            self.mask.clear()
        angles = range(0, 180, self.angle_step)
        self.mask = np.zeros((n_mask, self.mask_size, self.mask_size))
        calcul_masks(angles, self.mask_size, self.mask)















