import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # To do
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.transpose(sobel_filter_x)

    return sobel_filter_x, sobel_filter_y


def filter_image(im, filter):
    # To do
    surroundedzero = np.pad(im, 1)
    # im_filtered = [[0 for _ in range(len(im))] for _ in range(len(im[0]))]
    im_filtered = np.zeros((im.shape[0], im.shape[1]))
    direction = [-1, 0, 1]
    for row in range(1, len(surroundedzero) - 1):
        for col in range(1, len(surroundedzero[0]) - 1):
            current_sum = 0
            for i in direction:
                for j in direction:
                    current_sum += surroundedzero[row + i][col + j] * filter[i + 1][j + 1]
            im_filtered[row - 1][col - 1] = current_sum

    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    # grad_mag = [[0 for _ in range(len(im_dx))] for _ in range(len(im_dx[0]))]
    # grad_angle = [[0 for _ in range(len(im_dx))] for _ in range(len(im_dx[0]))]
    grad_mag = np.zeros((im_dx.shape[0], im_dx.shape[1]))
    grad_angle = np.zeros((im_dx.shape[0], im_dx.shape[1]))

    for row in range(len(im_dx)):
        for col in range(len(im_dx[0])):
            grad_mag[row][col] = np.sqrt(im_dx[row][col] ** 2 + im_dy[row][col] ** 2)
            if np.arctan2(im_dy[row][col], im_dx[row][col]) < 0:
                angle = np.arctan2(im_dy[row][col], im_dx[row][col]) + np.pi
            else:
                angle = np.arctan2(im_dy[row][col], im_dx[row][col])
            grad_angle[row][col] = angle

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    bin_size = 6
    M = len(grad_mag) // cell_size
    N = len(grad_mag[0]) // cell_size
    ori_histo = np.zeros((M, N, bin_size))

    direction = [i for i in range(cell_size)]
    for i in range(M):
        for j in range(N):
            for x in direction:
                for y in direction:

                    current_mag = grad_mag[i * cell_size + x][j * cell_size + y]
                    current_angle = grad_angle[i * cell_size + x][j * cell_size + y] * 180 / np.pi

                    if 0 <= current_angle < 15 or 165 <= current_angle < 180:
                        ori_histo[i][j][0] += current_mag

                    elif 15 <= current_angle < 45:
                        ori_histo[i][j][1] += current_mag

                    elif 45 <= current_angle < 75:
                        ori_histo[i][j][2] += current_mag

                    elif 75 <= current_angle < 105:
                        ori_histo[i][j][3] += current_mag

                    elif 105 <= current_angle < 135:
                        ori_histo[i][j][4] += current_mag

                    elif 135 <= current_angle < 165:
                        ori_histo[i][j][5] += current_mag

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    bin_size = 6
    row_num = len(ori_histo) - (block_size - 1)
    col_num = len(ori_histo[0]) - (block_size - 1)
    depth = bin_size * block_size * block_size
    ori_histo_normalized = np.zeros((row_num, col_num, depth))

    direction = [i for i in range(block_size)]
    for row in range(row_num):
        for col in range(col_num):
            block_dep = 0
            curr_sum = 0
            for x in direction:
                for y in direction:
                    for dep in range(bin_size):
                        ori_histo_normalized[row][col][block_dep] = ori_histo[row + x][col + y][dep]
                        curr_sum += ori_histo[row + x][col + y][dep] ** 2
                        block_dep += 1

            curr_sum = np.sqrt(curr_sum + 0.0001 ** 2)
            for i in range(depth):
                ori_histo_normalized[row][col][i] /= curr_sum

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    # im
    # #Normalize
    # im_mean = np.mean(im, axis=0)
    # im_variance = np.std(im, axis=0) + 1e-9
    # im = (im - im_mean) / im_variance
    cell_size = 8
    block_size = 2

    filter_x, filter_y = get_differential_filter()
    im_filtered_x = filter_image(im, filter_x)
    im_filtered_y = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_filtered_x, im_filtered_y)
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    hog = get_block_descriptor(ori_histo, block_size)

    hog = hog.reshape((-1))
    # visualize to verify


    plt.imshow(im_filtered_x, cmap='cool', interpolation='nearest')
    plt.show()
    plt.imshow(im_filtered_y, cmap='cool', interpolation='nearest')
    plt.show()
    plt.imshow(grad_mag, cmap='cool', interpolation='nearest')
    plt.show()
    plt.imshow(grad_angle, cmap='cool', interpolation='nearest')
    plt.show()

    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size ** 2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized ** 2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi / num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size * num_cell_w: cell_size],
                                 np.r_[cell_size: cell_size * num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_extract_Hog(im):
    im = im.astype('float') / 255.0
    # To do
    cell_size = 8
    block_size = 2

    filter_x, filter_y = get_differential_filter()
    im_filtered_x = filter_image(im, filter_x)
    im_filtered_y = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_filtered_x, im_filtered_y)
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    hog = get_block_descriptor(ori_histo, block_size)

    return hog


def IoU_func(firstBox, secondBox, rowlength, collength):
    x_diff = abs(firstBox[0] - secondBox[0])
    y_diff = abs(firstBox[1] - secondBox[1])

    overlapped_area = (rowlength - x_diff) * (collength - y_diff)
    IoU = overlapped_area / (2 * rowlength * collength - overlapped_area)
    return IoU


def non_maximum_suppression(similarity_record, rowlength, collength):
    similarity_record.sort(key=lambda x: x[2],reverse = True)

    returnposition = []
    returnposition.append(similarity_record[0])
    #similarity_record = np.array(similarity_record).reshape(len(similarity_record),3)
    #returnposition = np.ones((len(similarity_record),3))

    Index = 0


    while (Index < len(returnposition)):

        returnposition = returnposition[0:Index+1]

        for i in range(Index+1,len(similarity_record)):
            if IoU_func(returnposition[Index], similarity_record[i], rowlength, collength) < 0.5:
                returnposition.append(similarity_record[i])

        Index += 1
        similarity_record = returnposition

    return returnposition


def similarity(I_template, I_target):
    hog_template = face_extract_Hog(I_template)
    hog_template = hog_template.reshape((-1))
    hog_template_mean = np.mean(hog_template)
    hog_template -= hog_template_mean

    unitlength_template = np.linalg.norm(hog_template)

    row_constraint = len(I_target) - len(I_template) + 1
    col_constraint = len(I_target[0]) - len(I_template[0]) + 1
    similarity_record = []

    for row in range(row_constraint + 1):
        for col in range(col_constraint + 1):
            I_target_cut = I_target[row:row + len(I_template), col:col + len(I_template)]
            I_target_cut_Hog = face_extract_Hog(I_target_cut)
            I_target_cut_Hog = I_target_cut_Hog.reshape((-1))
            I_target_cut_Hog -= np.mean(I_target_cut_Hog)
            dot_product = np.dot(hog_template, I_target_cut_Hog)

            NCC = dot_product / (np.linalg.norm(I_target_cut_Hog) * unitlength_template)
            if NCC >= 0.45:
                similarity_record.append([col, row, NCC])

    returnposition = non_maximum_suppression(similarity_record, len(I_template), len(I_template[0]))
    #return np.array(similarity_record)
    return np.array(returnposition)


def face_recognition(I_target, I_template):
    a = similarity(I_template, I_target)
    return a




def visualize_face_detection(I_target, bounding_boxes, box_size):
    hh, ww, cc = I_target.shape

    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww - 1:
            x1 = ww - 1
        if x2 < 0:
            x2 = 0
        if x2 > ww - 1:
            x2 = ww - 1
        if y1 < 0:
            y1 = 0
        if y1 > hh - 1:
            y1 = hh - 1
        if y2 < 0:
            y2 = 0
        if y2 > hh - 1:
            y2 = hh - 1
        fimg = cv2.rectangle(fimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 2], (int(x1) + 1, int(y1) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    # im = cv2.imread("C:/Users/Lucky/PycharmProjects/cv_hw1/cameraman.tif", 0)
    # hog = extract_hog(im)


    I_target = cv2.imread("C:/Users/Lucky/PycharmProjects/cv_hw1/nuan2.jpg", 0)
    #I_target = I_target[0:80,0:80]
    # MxN image

    I_template = cv2.imread("C:/Users/Lucky/PycharmProjects/cv_hw1/nuan1.jpg", 0)
    # mxn  face template
    #
    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c = cv2.imread("C:/Users/Lucky/PycharmProjects/cv_hw1/target.png")
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])


#
