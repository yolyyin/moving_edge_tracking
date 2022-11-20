import numpy as np
import cv2
import math
from me_utils import compute_delta, is_out_of_img
from me_site import MeSite


def show_and_store_clicks(event, x, y, flags, param):
    """
    cv2 mouse callback event to show and store clicks
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param[1].append((x, y))
        cv2.circle(param[0],(x,y),2,(255,255,0),-1)


def tukey_influence(norm_res,standard):
    C = standard*4.6851
    vec = norm_res/C
    weights = np.where(np.square(vec)>1, 0, np.square(1-np.square(vec)))
    return weights


def partition(array,left,right):
    """
    partition algorithm in quicksort
    """
    r = right-1
    l = left
    value = array[right]

    while True:
        while array[r]>=value:
            r-=1
            if r < left:
                break
        while array[l]<value:
            l+=1
        if l >= r:
            break
        tmp = array[r]
        array[r]=array[l]
        array[l]=tmp
    tmp = array[right]
    array[right]=array[l]
    array[l]=tmp
    return l


def select(array, left, right, k):
    """
    select the element at kth position in the sorted array
    """
    while right>left:
        i = partition(array,left,right)
        if i >= k:
            right=i-1
        if i <= k:
            left = i+1
    return array[k]


class MeLine:
    def __init__(self, me):

        self.extrems = []
        self.delta = 0.0
        self.p_list = []
        self.sample_step = 0.0
        self.me_ = me
        self.track_cache = None

    def init_handpick_track(self, sample_step, image):
        """
        init tracking through hand picking the two extrem points of a line-segment
        """
        img_to_pick = image.copy()
        points=[]
        cv2.namedWindow('pick extrem points')
        cv2.setMouseCallback('pick extrem points', show_and_store_clicks, [img_to_pick, points])
        print("Pleas pick 2 points to define a line.....")
        while len(points) < 2:
            cv2.imshow('pick extrem points', img_to_pick)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        extrem0 = MeSite(self.me_, pic_point=points[0])
        extrem1 = MeSite(self.me_, pic_point=points[1])
        self.init_tracking(extrem0, extrem1, sample_step, image)


    def init_tracking(self, extrem0, extrem1, sample_step, image):
        """
        init tracking according to 2 line segment's extrem points and sample steps,
        now the sample is taken solely according to 2d images,
        3d sample idea is deprecated here
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.sample_step = sample_step
        self.extrems = [extrem0, extrem1]
        self.delta = compute_delta(extrem0.i, extrem0.j, extrem1.i, extrem1.j)
        self.sample(sample_step, image)
        self.init_conv(image)
        # update extrems data
        self.extrems[0].alpha = self.extrems[1].alpha = self.delta

    def init_conv(self, image):
        for me_point in self.p_list:
            me_point.update_conv_old(image)


    def sample(self, sample_step, image):
        img_width = image.shape[1]
        img_height = image.shape[0]

        # calculate 2d length for the line seg
        diff_i = self.extrems[1].i - self.extrems[0].i
        diff_j = self.extrems[1].j - self.extrems[0].j
        le_diff = np.sqrt(diff_i**2 + diff_j**2)

        if (le_diff <= np.finfo(float).eps):
            raise Exception("Extreme points too close for sampling ")
        n_sample = int(le_diff/sample_step) + 1

        # prepare step_vec, starting point for iteration
        step_i = diff_i/(n_sample-1)
        step_j = diff_j/(n_sample-1)
        cur_i = self.extrems[0].i
        cur_j = self.extrems[0].j
        self.p_list.clear()

        for n in range(n_sample):

            # if point is in the image, add to the sample list
            if not is_out_of_img(cur_i, cur_j, 0, img_width, img_height):
                me_point = MeSite(self.me_, pic_point=(cur_i, cur_j), alpha=self.delta)
                self.p_list.append(me_point)

            cur_i += step_i
            cur_j += step_j

        # update extreme
        self.extrems[0] = self.p_list[0]
        self.extrems[1] = self.p_list[-1]

        #print(f"\nMeLine Sample: {n_sample} points inserted in the list.\n")


    def resample(self, image):
        """
        resample the line if the number of sample is less than 90% of the
        expected value.
        the expected value = length of line//sample_step +1
        """
        # calculate 2d length for the line seg
        diff_i = self.extrems[1].i - self.extrems[0].i
        diff_j = self.extrems[1].j - self.extrems[0].j
        le_diff = np.sqrt(diff_i ** 2 + diff_j ** 2)

        expect_n = int(le_diff/self.sample_step) + 1

        if len(self.p_list) < 0.7 * expect_n:
            self.sample(self.sample_step, image)


    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.p_list:
            raise Exception("Tracking error: to few pixel to track")

        # loop through site list to track, use track_cache to cache pixel coordinates
        for i, me_point in enumerate(self.p_list):
            if me_point.state == 0:
                try:
                    me_point.track(image, test_contrast=False)
                except Exception as e:
                    print(f"\nTracking error: cannot track point {i}, error: {e}\n")



        self.suppress_points()
        self.set_extrems()

        self.seek_extrems(image, self.sample_step)
        self.least_square()

        self.suppress_points()
        self.set_extrems()

        self.resample(image)
        self.update_list_delta()



    def suppress_points(self):
        for me_point in self.p_list:
            if me_point.state != 0:  # suppresion state
                self.p_list.remove(me_point)

    def set_extrems(self):
        """
        decide and set extremity points among all the point in the point list
        """
        i_min = float('inf')
        j_min = float('inf')
        i_max = -1.
        j_max = -1.
        me_min = None
        me_max = None
        for me_point in self.p_list:
            if me_point.i < i_min:
                i_min = me_point.i
                j_min = me_point.j
                me_min = me_point
            if me_point.i > i_max:
                i_max = me_point.i
                j_max = me_point.j
                me_max = me_point

        # if i_min is about equal to i_max, which means the line is roughly vertical
        # turn to j to decide extrems
        if math.fabs(i_min - i_max) < 25:
            for me_point in self.p_list:
                if me_point.j < j_min:
                    j_min = me_point.j
                    me_min = me_point
                if me_point.j > j_max:
                    j_max = me_point.j
                    me_max = me_point

        if (me_min != None) and (me_max != None):
            self.extrems[0] = me_min
            self.extrems[1] = me_max
        else:
            raise Exception("\nError track result: Either the points list is empty or extrems are undefined! \n")

    def seek_extrems(self, image, step):
        """
        extrapolate qualified point to the line.
        the method is to test 3 extrapolation point further from current extremity points
        at equal intervals, track them with the same convolution mask,
        being trackable means the linear gradient on the specific angle is big enough
        around the point, and it should be on the line with the same angle.
        notice continuity of the line is not guaranteed
        """
        img_height = image.shape[0]
        img_width = image.shape[1]

        if step == 0:
            raise Exception("cannot seek_extrems with step=0")
        # calculate 2d length for the line seg
        diff_i = self.extrems[1].i - self.extrems[0].i
        diff_j = self.extrems[1].j - self.extrems[0].j
        le_diff = np.sqrt(diff_i ** 2 + diff_j ** 2)
        if (le_diff <= np.finfo(float).eps):
            raise Exception("Extreme points too close for sampling ")
        n_sample = int(le_diff / step) + 1 #

        # calculate temp

        # prepare step_i, step_j, starting point for iteration
        step_i = diff_i / (n_sample - 1)
        step_j = diff_j / (n_sample - 1)

        cur_i = self.extrems[0].i
        cur_j = self.extrems[0].j
        n_new = 0
        # look into 3 points further from extrems[0] for new extrems
        for n in range(3):
            # seek for points further from extrems[0]
            cur_i -= step_i
            cur_j -= step_j

            # if point is in the image, track the point,
            # if it's trackable the point will move to the places where the conv value
            # for line delta is the biggest, just as other points on the line
            if not is_out_of_img(cur_i, cur_j, 5, img_width, img_height):
                me_point = MeSite(self.me_, pic_point=(cur_i, cur_j), alpha=self.delta)
                me_point.track(image, False)
                if me_point.state == 0:  # NO_SUPPRESSION state
                    self.p_list.insert(0, me_point)
                    n_new += 1

        cur_i = self.extrems[1].i
        cur_j = self.extrems[1].j
        # look into 3 points further from extrems[1] for new extrems
        for n in range(3):
            # seek for points further from extrems[1]
            cur_i += step_i
            cur_j += step_j
            if not is_out_of_img(cur_i, cur_j, 0, img_width, img_height):
                me_point = MeSite(self.me_, pic_point=(cur_i, cur_j), alpha=self.delta)
                me_point.track(image, False)
                if me_point.state == 0:  # NO_SUPPRESSION state
                    self.p_list.append(me_point)
                    n_new += 1

        #print(f"\nFound {n_new} new extrems.\n")
        #print(f"\nMeLine Sample: {n_sample + n_new} points inserted in the list.\n")


    def send_paras_to_optimizer(self):
        point_pixels = np.zeros((len(self.p_list), 2))
        world_pixels = np.zeros((len(self.p_list), 3))

        for n, me_point in enumerate(self.p_list):
            point_pixels[n, 0] = me_point.i
            point_pixels[n, 1] = me_point.j
            world_pixels[n] = np.array([me_point.world_co[0], me_point.world_co[1], 0])

        self.me_.point_pixels.append(point_pixels)
        self.me_.world_pixels.append(world_pixels)
        self.me_.world_co_ys.append(self.extrems[0].world_co[1])


    def least_square(self):
        """
        purely 2d least_square algorithm,compute the approximated line seg,
        set suppressive state to outliers from this line
        """
        num_points = len(self.p_list)
        temp_delta= compute_delta(self.extrems[0].i, self.extrems[0].j,
                                  self.extrems[1].i, self.extrems[1].j)
        point_pixels = np.zeros((num_points, 2))

        for n, me_point in enumerate(self.p_list):
            point_pixels[n, 0] = me_point.i
            point_pixels[n, 1] = me_point.j

        if math.fabs(math.sin(temp_delta))< 0.5:
            """
            the line is far from vertical, use wi+b=j to construct Ax=B, where
            row of A is [i,1], row of B is [j], x =transpose([w,b])
            """
            A = np.hstack([point_pixels[:, 0:1], np.ones((num_points, 1))])
            B = point_pixels[:, 1]
            A_pseudo_inv = np.linalg.pinv(A)
            #A_pseudo_inv = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A))
            x = A_pseudo_inv.dot(B)
            residuals = B - A.dot(x)
            self.delta = math.atan(x[0])
        else:
            """
            the line is close to vertical, use wj+b=i to construct Ax=B, where
            row of A is [j,1], row of B is [i], x =transpose([w,b])
            """
            A = np.hstack([point_pixels[:, 1:2], np.ones((num_points, 1))])
            B = point_pixels[:, 0]
            A_pseudo_inv = np.linalg.pinv(A)
            # A_pseudo_inv = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A))
            x = A_pseudo_inv.dot(B)
            residuals = B - A.dot(x)
            self.delta = math.atan2(1, x[0])

        # normalize residuals by its median
        mid_pos = int(num_points/2.)
        sorted_res = residuals.copy()
        # find the median of the residual with quicksort
        median = select(sorted_res,0,num_points-1,mid_pos)

        norm_residuals = np.fabs(residuals-np.ones((num_points,))*median)
        # dif(residual - median)
        norm_sort_residuals = np.fabs(sorted_res-np.ones((num_points,))*median)
        # find the median of dif(residual-median), use it as a standard for outlier
        standard = select(norm_sort_residuals,0,num_points-1,mid_pos)
        standard *= 1.4826
        standard = max(standard,0.0001) # 0.001 is a minimum threshold here
        weights = tukey_influence(norm_residuals,standard)

        for n, me_point in enumerate(self.p_list):
            #TODO: need research here, use median to determine if suppress the point
            if me_point.state == 0 and weights[n] < 0.25:
                me_point.state = 3 # LS_SUPPRESSION

    def update_delta(self):
        self.delta = compute_delta(self.extrems[0].i, self.extrems[0].j, self.extrems[1].i, self.extrems[1].j)

    def update_list_delta(self):
        for me_point in self.p_list:
            me_point.alpha = self.delta

    def display(self, image):
        for me_point in self.p_list:
            if me_point.state == 0: # no supression
                point = (int(me_point.i), int(me_point.j))
                image = cv2.circle(image, point, 2, (110, 255, 255), -1)
        return image
