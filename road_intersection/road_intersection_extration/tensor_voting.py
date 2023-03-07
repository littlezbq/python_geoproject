import time
import numpy as np
import cv2 as cv
import math
from skimage import measure
from PIL import Image
np.set_printoptions(precision=1000)
np.set_printoptions(suppress=True)


def ind2sub(array_shape, ind):
    rows = np.floor(ind.astype('int') % array_shape[0])
    cols = np.floor(ind.astype('int') / array_shape[0])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


def find(condition):
    res = np.nonzero(condition)
    return res


class TensorVoting:

    def __init__(self, img_path):
        """
        Read Images Of Road Already Extracted
        :param img_path:
        """
        img = cv.imread(img_path)
        # img = np.asarray(Image.open(img_path).convert('RGB'))
        self.img = cv.cvtColor(img, code=cv.COLOR_RGB2GRAY)

    def run(self):
        # Run the tensor voting framework
        # plt.subplot(2, 2, 1)
        # plt.imshow(self.img, cmap='gray')
        T, ball = self.find_features(self.img, 18)

        #  Threshold un-important data that may create noise in the output.
        e1, e2, l1, l2 = self.convert_tensor_ev(T)
        #
        z = l1 - l2
        #
        # 归一化
        z = z / z[np.unravel_index(np.argmax(z, axis=None), z.shape)]
        #
        # # ret, _ = threshold_with_otsu(z)
        l1[z < 0.3] = 0
        l2[z < 0.3] = 0
        #
        T = self.convert_tensor_ev(e1, e2, l1, l2)
        #
        # plt.subplot(2, 2, 2)
        # plt.imshow(z, cmap='gray')
        # # subplot(2, 2, 2), imshow(l1 - l2, 'DisplayRange', [min(z(:)), max(z(:))]);
        #
        # # Run a local maxima algorithm on it to extract curves
        # re = self.calc_ortho_extreme(T, 15, np.pi / 8)
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(re, cmap='gray')
        #
        # # 画出球向量显著性图
        # plt.subplot(2, 2, 4)
        # plt.imshow(ball, cmap='gray')
        #
        # plt.show()

        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
        # ax0.imshow(ball, plt.cm.gray)
        # ax1.imshow(ball, plt.cm.gray)

        # Find Shape
        contours = measure.find_contours(ball, 0.5)

        # for n, contour in enumerate(contours):
        #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=1)
        #
        # ax1.axis('image')
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # plt.show()

        # Turn shape into points
        middle_point = [np.average(contour,axis=0) for contour in contours]

        # Draw points into original image
        self.img[self.img > 0] = 255
        img = np.array(Image.fromarray(self.img).convert("RGB"))

        # 创建一个黑色的单通道图像
        backgrond = np.array(Image.new('L', (self.img.shape[1], self.img.shape[0]), 0))

        for middle in middle_point:
            cv.circle(img,(int(middle[1]), int(middle[0])),10,(0,0,255),-1)
            cv.circle(backgrond,(int(middle[1]), int(middle[0])),10,255,-1)

        return img,backgrond

    def find_features(self, img, sigma):
        """
            FIND_FEATURES returns the tensorfield after voting on the binary image im
            using the sigma supplied.

            T = find_features(im,sigma)

            IM should be a binary (logical) image.
            SIGMA should be the scale used in voting, i.e. 18.25.

            Returns a tensor field T
        :param sigma: voting scale
        :return: T
        """

        """
            Calculate cached voting field at various angles, this way we can save
            a lot of time by preprocessing this data.
        """
        cached_vtf = self.create_cached_vf(sigma)

        """
            normalize the gray scale image from 0 to 1
        """
        img = img / img[np.unravel_index(np.argmax(img, axis=None), img.shape)]

        """
             First step is to produce the initially encode the image
             as sparse tensor tokens.
        """
        sparse_tf = self.calc_sparse_field(img)

        """
            First run of tensor voting, use ball votes weighted by
            the images grayscale.
        """
        refined_tf = self.calc_refined_field(sparse_tf, img, sigma)

        """
            third run is to apply the stick tensor voting after
            zero'ing out the e2(l2) components so that everything
            is a stick vote.
        """
        e1, e2, l1, l2 = self.convert_tensor_ev(refined_tf)

        """
            Get Ball Saliency
        """
        l2_ = np.multiply(l2, -1)
        ball = l2_ / l2_[np.unravel_index(np.argmax(l2_, axis=None), l2_.shape)]  # 缩放到0-1之间

        # 简单滤波, 0.7为模拟阈值，后续必须改为自适应阈值

        # ret, _ = threshold_with_otsu(ball)
        #
        # # 将阈值重新变回0-1之间的值
        # ret = ret  / 255
        #
        threshold = 0.65
        ball[ball < threshold] = 0
        ball[ball > threshold] = 1

        l2[:] = 0

        zerol2_tf = self.convert_tensor_ev(e1, e2, l1, l2)

        T = self.calc_vote_stick(zerol2_tf, sigma, cached_vtf)

        return T, ball

    def create_cached_vf(self, sigma):
        ws = np.floor(np.ceil(np.sqrt(-np.log(0.01) * sigma ** 2) * 2) / 2) * 2 + 1
        out = np.zeros([180, int(ws), int(ws), 2, 2])

        # num_list = [i for i in range(1, 181)]
        #
        # x = list(map(lambda num: np.cos(np.pi / 180 * num), num_list))
        # y = list(map(lambda num: np.sin(np.pi / 180 * num), num_list))
        st = time.time()
        for i in range(1, 181):
            x = np.cos(np.pi / 180 * i)
            y = np.sin(np.pi / 180 * i)
            v = [x, y]
            Fk = self.create_stick_tensorfield(v, sigma)

            # e1,e2,l1,l2 = self.convert_tensor_ev(Fk)
            out[i - 1, :, :, :, :] = Fk
        ed = time.time()
        print("Creating cached_vf in {:.2f} seconds".format(ed - st))

        return out

    def create_stick_tensorfield(self, uv, sigma):
        """
            CREATE_STICK_TENSORFIELD Creates a second order tensor
            field aligned along the unit vector provided and with
            the scale of sigma.

            unit_vector describes the direction the field should go in
            it's a 2 column, 1 row matrix where the first element is
            the x axis, the second element is the y axis. Default is
            [1,0], aligned on the x-axis.

            sigma describes the scale for the tensorfield, default is
            18.25.

            Ret urns T a MxMx2x2 tensor field, where M is the rectangular
            size of the field.

            Example:
                T = create_stick_tensorfield(unit_vector, sigma);

        :return:
        """

        """
            Generate initial parameters used by the entire system

            Calculate the window size from sigma using 
            equation 5.7 from Emerging Topics in Computer Vision
            make the field odd, if it turns out to be even.
        """
        ws = int(np.floor(np.ceil(np.sqrt(-np.log(0.01) * sigma ** 2) * 2) / 2) * 2 + 1)
        whalf = int((ws - 1) // 2)

        #  Turn the unit vector into a rotation matrix
        rot = np.array([[(uv[0] / np.hypot(uv[0], uv[1])), (-uv[1] / np.hypot(uv[0], uv[1]))],
                        [(uv[1] / np.hypot(uv[0], uv[1])), (uv[0] / np.hypot(uv[0], uv[1]))]])
        btheta = np.arctan2(uv[1], uv[0])

        '''
            Generate our theta's at each point in the
            field, adjust by our base theta so we rotate
            in funcion.
        '''
        a = np.array([i for i in range(-whalf, whalf + 1, 1)])
        b = np.array([i for i in range(whalf, -whalf - 1, -1)])
        X, Y = np.meshgrid(a, b)
        Z = np.dot(rot.T.conjugate(),
                   np.concatenate([np.reshape(X, (-1, 1), 'F'), np.reshape(Y, (-1, 1), 'F')], axis=1).T)

        X = np.reshape(Z[0, :], (ws, ws), 'F')
        Y = np.reshape(Z[1, :], (ws, ws), 'F')

        theta = np.arctan2(Y, X)

        '''Generate the tensor field direction aligned with the normal'''
        Tb = np.reshape(np.concatenate([theta, theta, theta, theta], axis=1), (ws, ws, 2, 2), 'F')  # 按列reshape
        T1 = -np.sin((2 * Tb + btheta), order='F')
        T2 = np.cos((2 * Tb + btheta), order='F')
        T3 = T1
        T4 = T2
        T1[:, :, 1, 0:3] = 1
        T2[:, :, 0:3, 0] = 1
        T3[:, :, 0:3, 1] = 1
        T4[:, :, 0, 0:3] = 1
        T = np.multiply(np.multiply(np.multiply(T1, T2, order='F'), T3, order='F'), T4, order='F')

        e1, e2, l1, l2 = self.convert_tensor_ev(T)

        '''
            Generate the attenuation field, taken from Equation
            5.2 in Emerging Topics in Computer Vision. Note our
            thetas must be symmetric over the Y axis for the arc
            length to be correct so there's a bit of a coordinate
            translation.
        '''

        theta = abs(theta)
        theta[theta > math.pi / 2] = math.pi - theta[theta > math.pi / 2]
        theta = 4 * theta

        s = np.zeros((ws, ws))
        k = np.zeros((ws, ws))

        '''Calculate the attenuation field.'''
        l = np.sqrt((np.power(X, 2, order='F') + np.power(Y, 2, order='F')), order='F')
        c = (-16 * np.log2(0.1) * (sigma - 1)) / np.pi ** 2

        aa1, aa2 = l != 0, theta != 0
        cond1 = aa1 & aa2

        s[cond1] = np.divide(
            np.multiply(theta[cond1], l[cond1], order='F'), np.sin(theta[cond1], order='F'), order='F')

        aa3, aa4 = l == 0, theta == 0
        cond2 = aa3 | aa4
        s[cond2] = l[cond2]
        k[l != 0] = np.divide(2 * np.sin(theta[l != 0], order='F'), l[l != 0], order='F')

        DF = np.exp(-((np.power(s, 2, order='F') + c * np.power(k, 2, order='F')) / (sigma ** 2)), dtype=np.float64,
                    order='F')
        DF[theta > (np.pi / 2)] = 0

        '''   Generate the final tensor field'''
        T = np.multiply(T, np.reshape(np.concatenate([DF, DF, DF, DF], axis=1), (ws, ws, 2, 2), order='F'))

        return T

    def convert_tensor_ev(self, *args):
        '''
        特征值分解
        :param T:
        :return:
        '''

        if len(args) == 1:
            i1 = args[0]
            K11 = i1[:, :, 0, 0]
            K12 = i1[:, :, 0, 1]
            K21 = i1[:, :, 1, 0]
            K22 = i1[:, :, 1, 1]

            n, p = K11.shape

            o1 = np.zeros([n, p, 2])
            o2 = np.zeros([n, p, 2])
            o3 = np.zeros([n, p])
            o4 = np.zeros([n, p])

            # trace / 2
            t = (K11 + K22) / 2

            a = K11 - t
            b = K12

            ab2 = np.sqrt((np.power(a, 2, order='F') + np.power(b, 2, order='F')), order='F')
            o3 = ab2 + t
            o4 = -ab2 + t

            theta = np.arctan2((ab2 - a), b)

            o1[:, :, 0] = np.cos(theta)
            o1[:, :, 1] = np.sin(theta)
            o2[:, :, 0] = -np.sin(theta)
            o2[:, :, 1] = np.cos(theta)

            return o1, o2, o3, o4
        else:
            i1, i2, i3, i4 = args[0], args[1], args[2], args[3]
            o1 = np.zeros([i3.shape[0], i3.shape[1], 2, 2])
            o1[:, :, 0, 0] = np.multiply(i3, np.power(i1[:, :, 0], 2, order='F'), order='F') + np.multiply(i4, np.power(
                i2[:, :, 0], 2, order='F'), order='F')
            o1[:, :, 0, 1] = np.multiply(np.multiply(i3, i1[:, :, 0], order='F'), i1[:, :, 1], order='F') + np.multiply(
                np.multiply(i4, i2[:, :, 0], order='F'), i2[:, :, 1], order='F')
            o1[:, :, 1, 0] = o1[:, :, 0, 1]
            o1[:, :, 1, 1] = np.multiply(i3, np.power(i1[:, :, 1], 1)) + np.multiply(i4, np.power(i2[:, :, 1], 1))

            return o1

    def calc_sparse_field(self, img):
        '''

        :param img:
        :return:
        '''

        h, w = img.shape
        T = np.zeros([h, w, 2, 2])

        rows, cols = np.where(img > 0)

        # 先按列排列转化成列下标
        I = [cols[i] * img.shape[0] + rows[i] for i in range(rows.shape[0])]
        I.sort()

        # 将I转化为Array
        I = np.reshape(I, (-1, 1), 'F')
        # 再分解成按列排布的行和列坐标
        rows = I % img.shape[0]
        cols = np.ceil(I // img.shape[0]).astype(int)

        n = rows.shape[0]

        st = time.time()
        for i in range(0, n):
            T[rows[i], cols[i], :, :] = np.array([[1, 0], [0, 1]])

        ed = time.time()

        print("Calc sparse field using {:.2f} seconds".format(ed - st))

        # e1, e2, l1, l2 = self.convert_tensor_ev(T)

        return T

    def calc_refined_field(self, tf, im, sigma):
        # Get votes for ball
        st = time.time()
        print("Generating refined field......")
        ball_vf = self.calc_vote_ball(tf, im, sigma)

        #  % Erase anything that's not in the original image
        rows, cols = np.where(im == 0)
        s = rows.shape[0]

        # 先按列排列转化成列下标
        I = [cols[i] * im.shape[0] + rows[i] for i in range(rows.shape[0])]
        I.sort()

        # 将I转化为Array
        I = np.reshape(I, (-1, 1), 'F')
        # 再分解成按列排布的行和列坐标
        rows = I % im.shape[0]
        cols = np.ceil(I // im.shape[0]).astype(int)

        for i in range(1, s + 1):
            ball_vf[rows[i - 1], cols[i - 1], 0, 0] = 0
            ball_vf[rows[i - 1], cols[i - 1], 0, 1] = 0
            ball_vf[rows[i - 1], cols[i - 1], 1, 0] = 0
            ball_vf[rows[i - 1], cols[i - 1], 1, 1] = 0

        ed = time.time()
        print("calc_refined_field using {:.2f} seconds".format(ed - st))

        return tf + ball_vf

    def calc_vote_ball(self, T, im, sigma):
        st = time.time()
        print("Ball Voting......")
        Fk = self.create_ball_tensorfield(sigma)

        # e1, e2, l1, l2 = self.convert_tensor_ev(Fk)

        wsize = int(np.floor(np.ceil(np.sqrt(-np.log(0.01) * sigma ** 2) * 2) / 2) * 2 + 1)
        wsize_half = int((wsize - 1) // 2)

        '''
             resize the tensor to make calculations easier. This gives us a margin
             around the tensor that's as large as half the window of the voting
             field so when we begin to multiple and add the tensors we dont have
             to worry about trimming the voting field to avoid negative and
             overflow array indices.
        '''
        Th = T.shape[0]
        Tw = T.shape[1]

        Tn = np.zeros([int(Th + wsize_half * 2), int(Tw + wsize_half * 2), 2, 2])
        Tn[wsize_half: (wsize_half + Th), wsize_half: (wsize_half + Tw), :, :] = T[0:, 0:, :, :]
        T = Tn

        # perform eigen-decomposition, assign default tensor from estimate
        e1, e2, l1, l2 = self.convert_tensor_ev(T)

        # Find everything that's above 0

        rows, cols = np.where(l1 > 0)

        I = [cols[i] * l1.shape[0] + rows[i] for i in range(rows.shape[0])]
        I.sort()

        # Loop through each stick found in the tensor T

        u, v = ind2sub(l1.shape, np.array(I).reshape(1, -1))
        p = u.shape[1]
        D = np.zeros([2, p])
        D[0, :] = u
        D[1, :] = v

        for i in range(D.shape[1]):
            # the current intensity of the vote Apply weights
            s = D[:, i]
            Zk = np.dot(im[int(s[0] - wsize_half), int(s[1] - wsize_half)], Fk)
            # Calculate positions of window, add subsequent values back in
            beginy = int(s[0] - wsize_half)
            endy = int(s[0] + wsize_half)
            beginx = int(s[1] - wsize_half)
            endx = int(s[1] + wsize_half)

            T[beginy: endy + 1, beginx: endx + 1, :, :] = T[beginy: endy + 1, beginx: endx + 1, :, :] + Zk

            # e1, e2, l1, l2 = self.convert_tensor_ev(T)
            #
            # op += 1
        # Trim the T of the margins we used.
        T = T[wsize_half:(wsize_half + Th), wsize_half: (wsize_half + Tw), :, :]

        # e1, e2, l1, l2 = self.convert_tensor_ev(T)
        ed = time.time()

        print("calc_vote_ball using {:.2f} seconds".format(ed - st))
        return T

    def create_ball_tensorfield(self, sigma):
        """
            CREATE_BALL_TENSORFIELD creates a ball tensor field, sigma
               determines the scale and size of the tensor field.

               Default for sigma is 18.25.
        :param sigma:
        :return:
        """

        wsize = int(np.floor(np.ceil(np.sqrt(-np.log(0.01) * sigma ** 2) * 2)))
        wsize = int(np.floor(wsize // 2) * 2 + 1)

        T = np.zeros([wsize, wsize, 2, 2])
        st = time.time()
        for theta in np.arange(0, 1 / 32 + 1 - 1 / 32, 1 / 32):
            theta = theta * 2 * np.pi

            v = list([np.cos(theta), np.sin(theta)])
            B = self.create_stick_tensorfield(v, sigma)

            # e1, e2, l1, l2 = self.convert_tensor_ev(B)

            T = T + B

        ed = time.time()

        print("Create ball tensorfield using {:.2f} seconds".format(ed - st))

        T = T / 32

        return T

    def calc_vote_stick(self, T, sigma, cachedvf):
        """
        CALC_VOTE_STICK votes using stick tensors returning a new
           tensor field.

           T = calc_vote_stick(M,sigma,cachedvf);

           M is the input tensor field, this should be an estimate.
           sigma determines the scale of the tensor field.
           cachedvf is a cached voting field produced by
           create_cached_vf in order to speed up voting process.
        :param T:
        :param sigma:
        :param cachedvf:
        :return:
        """
        st = time.time()
        print("Stick Voting......")
        wsize = int(np.floor(np.ceil(np.sqrt(-np.log(0.01) * sigma ** 2) * 2) / 2) * 2 + 1)
        wsize_half = int((wsize - 1) // 2)

        '''
             resize the tensor to make calculations easier. This gives us a margin
             around the tensor that's as large as half the window of the voting
             field so when we begin to multiple and add the tensors we dont have
             to worry about trimming the voting field to avoid negative and
             overflow array indices.
        '''

        Th = T.shape[0]
        Tw = T.shape[1]

        Tn = np.zeros([Th + wsize_half * 2, Tw + wsize_half * 2, 2, 2])
        Tn[(wsize_half): (wsize_half + Th), (wsize_half): (wsize_half + Tw), :, :] = T[0:, 0:, :, :]
        T = Tn

        # perform eigen-decomposition, assign default tensor from estimate

        e1, e2, l1, l2 = self.convert_tensor_ev(T)

        # Find everything that's a stick vote.  Prehaps use a threshold here?
        rows, cols = np.where(l1 > 0)

        I = [cols[i] * l1.shape[0] + rows[i] for i in range(rows.shape[0])]
        I.sort()

        # Loop through each stick found in the tensor T.
        a = 0

        u, v = ind2sub(l1.shape, np.array(I).reshape(1, -1))
        p = u.shape[1]
        D = np.zeros([2, p])
        D[0, :] = u
        D[1, :] = v
        op = int(np.ceil(p * 0.01))

        args = self.calc_vote_stick.__code__.co_argcount - 1

        for i in range(D.shape[0]):
            # the direction is e1 with intensity l1-l2
            s = D[:, i]
            v = e1[int(s[0]), int(s[1]), :]

            if args < 6:
                v = v[:]
                Fk = self.create_stick_tensorfield([-v[1], v[0]], sigma)
            else:
                angle = np.round(180 / np.pi * math.atan(v[1] / v[0]))
                if angle < 1:
                    angle = angle + 180
                # Fk = shiftdim(cachedvf(angle,:,:,:,:))

            # the current intensity of the vote
            # Apply weights

            Fk = (l1[int(s[0]), int(s[1])] - l2[int(s[0]), int(s[1])]) * Fk

            # Calculate positions of window, add subsequent values back in
            beginy = int(s[0] - wsize_half)
            endy = int(s[0] + wsize_half)
            beginx = int(s[1] - wsize_half)
            endx = int(s[1] + wsize_half)
            T[beginy: endy + 1, beginx: endx + 1, :, :] = T[beginy: endy + 1, beginx: endx + 1, :, :] + Fk

            # Trim the T of the margins we used
            T = T[(wsize_half):(wsize_half + Th), (wsize_half): (wsize_half + Tw), :, :]

            ed = time.time()
            print("calc_vote_stick using {:.2f} seconds".format(ed - st))
            return T

    def calc_ortho_extreme(self, T, r, epsilon):
        """
           CALC_ORTHO_EXTREME examines the orthogonal or "normal" on the curve
           and finds the maxima along the line. R is the length to check
           along the normal axis. Epsilon is the width in radians to check,
           i.e. pi/8

        :param T:
        :param r:
        :param epsilon:
        :return:
        """
        st = time.time()
        print("calc_ortho_extreme begin......")
        e1, e2, l1, l2 = self.convert_tensor_ev(T)
        q = l1 - l2
        h, w = l1.shape
        re = np.zeros([h, w])

        a = np.array([i for i in range(-r, r + 1, 1)])
        b = np.array([i for i in range(-r, r + 1, 1)])

        X, Y = np.meshgrid(a, b)
        t = np.arctan2(Y, X)
        l = np.sqrt(np.power(X, 2) + np.power(Y, 2))
        q1 = np.zeros([int(2 * r + h), int(2 * r + w)])
        q1[int(r): int(h + r), int(r): int(w + r)] = q

        rows, cols = np.where(q1 > 0)

        D = [cols[i] * q1.shape[0] + rows[i] for i in range(rows.shape[0])]
        D.sort()

        D = np.array(D).reshape(-1, 1)
        h, w = q1.shape

        for i in range(0, len(D)):
            y, x = ind2sub(q1.shape, D[i, 0])
            X2 = np.multiply(l, np.cos(t + np.arctan2(e1[int(y - r), int(x - r), 1], e1[int(y - r), int(x - r), 0])))
            Y2 = np.multiply(l, np.sin(t + np.arctan2(e1[int(y - r), int(x - r), 1], e1[int(y - r), int(x - r), 0])))
            t2 = np.abs(np.arctan2(Y2, X2))
            t2[t2 > np.pi / 2] = np.pi - t2[t2 > np.pi / 2]

            t2[t2 <= epsilon] = 1
            t2[t2 != 1] = 0
            t2[l > r] = 0

            try:
                z = np.multiply(q1[int(y - r): int(y + r + 1), int(x - r): int(x + r + 1)], t2)
            except:
                pass

            z = (z > q1[int(y), int(x)]).max(0)
            if z.max() == 0:
                re[int(y - r - 1), int(x - r)] = 1

        ed = time.time()
        print("calc_ortho_extreme using {:.2f}".format(ed - st))
        return re


if __name__ == "__main__":
    img_path = "../datas/10828780_15.png"
    tvf = TensorVoting(img_path)
    tvf.run()
