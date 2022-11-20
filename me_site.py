import numpy as np
import math

from me_utils import is_out_of_img


class MeSite:
    def __init__(self, me, pic_point=None, alpha=0.0):

        self.i = self.j = 0.
        self.alpha = alpha
        self.conv_old = self.gradient_old = 0.0
        self.state = 0  # no suppression

        # private paras
        self.me_ = me

        if pic_point is not None:
            self.i, self.j = pic_point

    def get_query_list(self, image, range_):
        """
        get query points on the lines perpendicular to the old points
        """
        k_range = np.arange(-range_, range_+1)
        pixel_is = self.i - k_range * math.sin(self.alpha)
        pixel_js = self.j + k_range * math.cos(self.alpha)
        list_query_pixels = np.transpose(np.vstack((pixel_is, pixel_js)))

        return list_query_pixels

    def convolution(self, query_point, image):
        """
        calculate convolution between query points and the chosen angle mask
        that fit the origin point's line angle
        """
        height = image.shape[0]
        width = image.shape[1]
        i, j = round(query_point[0]), round(query_point[1])

        half_mask = self.me_.mask_size // 2  # TODO：!!此处需思考
        alpha = self.alpha  # copy origin point's alpha

        if is_out_of_img(i, j, half_mask + self.me_.strip, width, height):
            # this query point is abandoned
            conv_res = None
        else:
            # restrain alpha between (0, pi)
            while alpha < 0:
                alpha += math.pi
            while alpha >= math.pi:
                alpha -= math.pi
            # convert radians to degrees
            alpha_degree = math.degrees(alpha)
            index_mask = round(alpha_degree/self.me_.angle_step) % 180
            # calculate istart, jstart
            istart = i - half_mask
            iend = i + half_mask
            jstart = j - half_mask
            jend = j + half_mask

            patch = image[jstart:jend+1, istart:iend+1]
            conv_res = np.sum(self.me_.mask[index_mask] * patch)

        return conv_res

    def update_conv_old(self, image):
        self.conv_old = self.convolution((self.i, self.j), image)

    def track(self, image, test_contrast=True):
        """
        the basic idea is to query points alongside normal line crossing the point,
        the convolution mask based on the line angle is applied to the local window
        around the query point and the original point,
        if the query point is on the line, the convolution value should be similar
        """
        # get a list of query points along the normal line
        max_rank = -1
        max_conv = 0
        max_likelihood = 0
        list_query_pixels = self.get_query_list(image, self.me_.range)
        contrast = 0
        contrast_max = 1 + self.me_.mu2
        contrast_min = 1 - self.me_.mu1
        # array in which likelihood ratios will be stored
        likelihood = [0.] * (2*self.me_.range + 1)
        diff = 1e6
        for n in range(2*self.me_.range + 1):
            # apply convolution mask to the query point
            conv_res = self.convolution(list_query_pixels[n, :], image)
            if conv_res is None:
                continue
            """
            IMPORTANT: currently the test_constrast method doesn't work when an end point is met,
            don't use test_contrast right now
            
            test contrast method:
            luminance of reference pixel and potential correspondent pixel
            must be similar, hence the conv ratio should lay between,
            for instance, 0.5 and 1.5
            """
            if test_contrast:
                likelihood[n] = math.fabs(2 * conv_res)
                if likelihood[n] > self.me_.threshold:
                    contrast = conv_res/self.conv_old  # !!!
                    if (contrast > contrast_min) and (contrast < contrast_max)\
                        and (math.fabs(1-contrast) < diff):
                        diff = math.fabs(1-contrast)
                        max_conv = conv_res
                        #max_likelihood = likelihood[n]
                        max_rank = n
            else:
                likelihood[n] = math.fabs(2 * conv_res)
                if (likelihood[n] > max_likelihood) and (likelihood[n] > self.me_.threshold):
                    max_conv = conv_res
                    max_likelihood = likelihood[n]
                    max_rank = n

        if max_rank >= 0:
            # update point location and old convolution value
            self.i, self.j = list_query_pixels[max_rank, :]
            self.gradient_old = math.sqrt(math.fabs(max_conv))
            self.conv_old = max_conv
        else:
            # none of the query sites is better than the threshold
            self.gradient_old = 0.
            if math.fabs(contrast) > 0:
                self.state = 1  # contrast suppression
            else:
                self.state = 2  # threshold suppression