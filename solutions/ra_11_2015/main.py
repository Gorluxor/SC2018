from __future__ import print_function
import math
import time
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from solutions.ra_11_2015.test import test

# Global constants
ACC_FOR_VALID_POSITION = 0.8
ACC_FOR_VALID_CONTOUR = 0.90


Debug = False
DebugImageChannels = False
DebugPreProcessingImage = False
DebugLines = False
DebugContour = False
DebugCNN = False


def image_bin(image_gs):
    ret, image_bin_tmp = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin_tmp


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    output_image = cv2.copyMakeBorder(
        region,
        2,
        3,
        4,
        4,
        cv2.BORDER_CONSTANT,
        value=0
    )
    res_img = cv2.resize(output_image, (28, 28), interpolation=cv2.INTER_NEAREST)
    return res_img


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def distance_euclid(t1, t2):
    return math.sqrt(math.pow(t1[0] - t2[0], 2) + math.pow(t1[1]-t2[1], 2))


def reshape_data(input_data):
    input_data = input_data.flatten()
    return input_data.reshape(1, 28, 28, 1)


def normalize(x):
    length = math.sqrt(math.pow(x[0], 2) + math.pow(x[1], 2))
    if length == 0:
        length = 1
    x[0] = x[0]*1.0 / length
    x[1] = x[1]*1.0 / length
    return x


class History:
    def __init__(self, x, y, w, h, num, acc):
        self.w = w
        self.h = h
        self.position = []
        self.position.append([x, y, num, acc])
        self.already_added = False
        self.already_minus = False
        self.valid_positions = 1
        self.missing = 0
        self.intersected_add = False
        self.intersected_minus = False

    def get_x(self):
        return self.position[len(self.position)-1][0]

    def get_y(self):
        return self.position[len(self.position)-1][1]

    def vote_number(self, extra_num=-1):
        n = len(self.position)
        occ_list = [0] * 10
        for i in range(n):
            x, y, num, acc = self.position[i]
            if acc > ACC_FOR_VALID_POSITION:
                occ_list[num] += 1
        if extra_num != -1:
            occ_list[extra_num] += 1
        if True:
            print(str(occ_list))
        return winner(occ_list)

    def add_position(self, x, y, num, acc):
        if acc > ACC_FOR_VALID_POSITION:
            self.valid_positions += 1
        self.position.append([x, y, num, acc])

    def vec_movement(self):
        n = len(self.position)

        vec = np.array([0, 0])
        vec_normalizovan = np.array([0, 0])
        for i in range(0, n-1, 2):
            x, y, num, acc = self.position[i]
            x_next, y_next, num_next, acc_next = self.position[i+1]
            rez = get_vec_between_two_pos([x,y],[x_next, y_next])
            vec_normalizovan += normalize(rez)
            vec += rez

        vec[0] = round(vec[0])
        vec[1] = round(vec[1])
        return vec, vec_normalizovan

    def num_history(self):
        return self.valid_positions
        # return len(self.position)

    def __del__(self):
        del self.position


def get_vec_between_two_pos(prev, next):
    v1 = np.array(prev)
    v2 = np.array(next)
    return v2 - v1


DIFF = 8
CHECK_JUMP = 4

def intersect_happened(add, history):
    happened = False
    if history.valid_positions > 34:
        line_selected = None
        if add:
            line_selected = lines1
        else:
            line_selected = lines2

        for i in range(0, len(history.position) - CHECK_JUMP + 1 - CHECK_JUMP * DIFF, CHECK_JUMP):
            if happened:
                break
            for line_x_1, line_y_1, line_x_2, line_y_2 in line_selected:
                x, y, num, acc = history.position[i]
                w = history.w
                h = history.h
                x_jump, y_jump, num_jump, acc_jump = history.position[i+CHECK_JUMP-1]

                if happened:
                    break
                else:
                    happened1 = intersect((x+w,y),(x_jump+w, y_jump), (line_x_1, line_y_1), (line_x_2, line_y_2))
                    happened2 = intersect((x,y+h),(x_jump, y_jump+h), (line_x_1, line_y_1), (line_x_2, line_y_2))
                    happened = happened1 or happened2
        for i in range(len(history.position) - CHECK_JUMP + 1 - CHECK_JUMP * DIFF, len(history.position) - CHECK_JUMP + 1, CHECK_JUMP):
            if happened:
                break
            for line_x_1, line_y_1, line_x_2, line_y_2 in line_selected:
                x, y, num, acc = history.position[i]
                w = history.w
                h = history.h
                x_jump, y_jump, num_jump, acc_jump = history.position[i + CHECK_JUMP - 1]
                vec, mov = history.vec_movement()
                if happened:
                    break
                else:
                    happened1 = intersect((x + w , y), (x_jump + w + mov[0], y_jump + mov[1]), (line_x_1, line_y_1), (line_x_2, line_y_2))
                    happened2 = intersect((x , y + h), (x_jump + mov[0], y_jump + h + mov[1]), (line_x_1, line_y_1), (line_x_2, line_y_2))
                    happened = happened1 or happened2

    return happened


def check_whole(L, x, y, w, h):
    L = L[0]
    x1 = L[0]
    y1 = L[1]
    x2 = L[2]
    y2 = L[3]
    d1 = intersect((x1, y1), (x2, y2), (x, y), (x, y+h))
    d2 = intersect((x1, y1), (x2, y2), (x, y + h), (x + w, y+h))
    d3 = intersect((x1, y1), (x2, y2), (x + w, y), (x + w, y+h))
    d4 = intersect((x1, y1), (x2, y2), (x, y), (x+w, y))
    intersected = d1 | d2 | d3 | d4
    return intersected


def select_roi(image_orig, bin_image):
    global model

    if Debug and DebugPreProcessingImage:
        cv2.imshow("Before", bin_image)
        cv2.setWindowTitle("Before", "Before")

    cnt_img = bin_image.copy()
    cnt_img = dilate(cnt_img)
    cnt_img = erode(cnt_img)
    cnt_img = dilate(cnt_img)
    cnt_img = erode(cnt_img)
    img, contours, hierarchy = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    history_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # region = image_bin[y:y+h+1, x:x+w+1]  # original
        region = bin_image[y - 2:y + h + 2, x - 2:x + w + 2]
        area = cv2.contourArea(contour)

        special_case = area == 0 and 29 > h > 18
        if (18 < area < 800) or special_case:
            if Debug and h > 0 and w > 0 and DebugContour:
                cv2.imshow("contour", region)
                name = "Contour :(" + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + ") area:" + str(area)
                cv2.setWindowTitle("contour", name)
                cv2.resizeWindow("contour", 600, 50)
            if special_case and Debug:
                print("Special")
            resized = resize_region(region)
            prediction = model.predict(reshape_data(resized))
            mp = max(prediction[0])
            number = np.argmax(prediction)
            correlation = round(w/3)
            if 21 > h > 11-correlation and w < 22:
                #  dodato da ne sme da sece liniju
                if mp > ACC_FOR_VALID_CONTOUR:  #\
                        # and not check_whole(lines1, x, y, w, h) and not check_whole(lines2, x, y, w, h):
                    if Debug and DebugCNN:
                        cv2.imshow("Original", resized)
                        cv2.setWindowTitle("Original", str(number)+"with"+str(mp))
                        cv2.resizeWindow('Original', 600, 50)
                        cv2.waitKey(0)
                    a = History(x, y, w, h, number, mp)

                    history_array.append(a)
                    regions_array.append(resized)
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_orig, regions_array, history_array


lines1 = []
lines2 = []


def find_lines(color_img):
    green_img = color_img[:, :, 1]
    blue_img = color_img[:, :, 0]
    if Debug and DebugImageChannels and DebugLines:
        red_img = color_img[:, :, 2]

        cv2.imshow("Blue_image", blue_img)
        cv2.imshow("Red", red_img)
        cv2.imshow("Green", green_img)

    kernel = np.ones((3, 3), np.uint8)
    blue_img = cv2.erode(blue_img, kernel, iterations=1)
    blue_img = cv2.dilate(blue_img, kernel, iterations=1)
    green_img = cv2.erode(green_img, kernel, iterations=1)
    green_img = cv2.dilate(green_img, kernel, iterations=1)
    low_threshold = 200
    high_threshold = 255

    p_green = cv2.Canny(green_img, threshold1=low_threshold, threshold2=high_threshold)
    p_blue = cv2.Canny(blue_img, threshold1=low_threshold, threshold2=high_threshold)

    kernel_size = 5
    p_blue = cv2.GaussianBlur(p_blue, (kernel_size, kernel_size), 0)
    p_green = cv2.GaussianBlur(p_green, (kernel_size, kernel_size), 0)

    max_line_gap = 84
    min_line_length = 202

    global lines1
    lines1 = cv2.HoughLinesP(p_blue, rho=1, theta=np.pi / 180, threshold=100, minLineLength=min_line_length,maxLineGap=max_line_gap)
    global lines2
    lines2 = cv2.HoughLinesP(p_green, rho=1, theta=np.pi / 180, threshold=100, minLineLength=min_line_length,maxLineGap=max_line_gap)

    # maybe more selection rather then 0
    lines1 = lines1[0]
    lines2 = lines2[0]

    if DebugLines:
        for x1, y1, x2, y2 in lines1:
            cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        for x1, y1, x2, y2 in lines2:
            cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        plt.subplot(1, 1, 1)
        plt.title('Lines')
        plt.imshow(color_img)
        plt.show()
        time.sleep(10)


def ccw(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    return (cy-ay) * (bx-ax) > (by - ay) * (cx-ax)


def intersect(a, b, c, d):
    val = ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    return val


model = None
current_sum = 0
old_history = []
history = []
max_height = 0
max_width = 0

def process_new_history(color_img):  # history = history_to_process
    global current_sum
    global old_history
    global history

    old_ind_found = []
    if old_history is not None:
        for i in range(len(old_history)):  # find close numbers that are same
            if old_history[i].missing < 5:
                potential_numbers = []
                x = old_history[i].get_x()
                y = old_history[i].get_y()
                for j in reversed(range(len(history))):
                    if history[j].position[0][2] == old_history[i].position[0][2]:
                        newest_x, newest_y, num, acc = history[j].position[0]
                        distance = distance_euclid([newest_x, newest_y], [x, y])
                        if distance < 16:
                            if newest_x >= x and newest_y >= y:
                                potential_numbers.append([distance, j])

                if potential_numbers.__len__() > 0:  # if there are multiple
                    potential_numbers = sorted(potential_numbers, key=lambda k: k[0])
                    newest_x, newest_y, num, acc = history[potential_numbers[0][1]].position[0]
                    cv2.line(color_img, (x, y), (newest_x, newest_y), (255, 0, 0), 2)
                    old_history[i].add_position(newest_x, newest_y, num, acc)
                    del history[potential_numbers[0][1]]
                    old_history[i].missing = 0
                    old_ind_found.append(i)

        for i in range(len(old_history)):  # try to find it as a missing number
            potential_numbers = []
            if not old_ind_found.__contains__(i) and old_history[
                i].valid_positions > 3 and old_history[i].missing < 100:
                x = old_history[i].get_x()
                y = old_history[i].get_y()
                vec, mov = old_history[i].vec_movement()
                done = False
                for k in range(1, old_history[i].missing + 21, 1):
                    x += mov[0]
                    y += mov[1]
                    if x > max_width:
                        done = True
                    if y > max_height:
                        done = True
                    if done:
                        break
                    for j in range(len(history)):  # for j in reversed(range(len(history))):
                        newest_x, newest_y, num, acc = history[j].position[0]
                        # if intersect((x,y),(x+vec[0],y+vec[0]), lines1, lines1):
                        distance = distance_euclid([newest_x, newest_y], [x, y])
                        if distance < 15 and num == old_history[i].position[0][2]:
                            if newest_x >= x and newest_y >= y:
                                newest_x, newest_y, num, acc = history[j].position[0]
                                cv2.line(color_img, (old_history[i].get_x(), old_history[i].get_y()),
                                         (newest_x, newest_y),
                                         (0, 0, 255), 2)
                                # x_t = old_history[i].get_x()
                                # y_t = old_history[i].get_y()
                                # old_history[i].add_missing_vec(np.array([x_t, y_t]), np.array([newest_x, newest_y]))
                                old_history[i].add_position(newest_x, newest_y, num, acc)
                                del history[j]
                                old_history[i].missing = 0
                                old_ind_found.append(i)

                                done = True
                                break

        for i in range(len(old_history)):  # add missing factor
            if not old_ind_found.__contains__(i):
                old_history[i].missing += 1




    # new numbers this frame
    if history is not None:
        if Debug:
            print("Number of new numbers in frame:"+str(len(history)))
        for j in range(0, len(history), 1):
            old_history.append(history[j])
        del history
    return color_img



def calculate(list_videos):
    global model
    global current_sum
    global op_history
    global old_history
    global history
    global max_height
    global max_width

    # model = load_model('model.h5')

    student = 'RA 11/2015 Aleksandar Cvejic\n'
    txt = 'file \tsum\n'
    t_sum = []
    for current_video_count in range(0, len(list_videos), 1):

        frame_count, video_frames = get_all_frames(video=list_videos[current_video_count])

        image_color = video_frames[0]
        max_height = image_color.shape[0]
        max_width = image_color.shape[1]

        find_lines(image_color.copy())  # GLOBAL PARAMETERS lines1 and lines2 set

        old_history = []
        current_sum = 0

        for j in range(0, frame_count, 3):

            current_frame = video_frames[j]
            one_channel = current_frame.copy()[:, :, 2]  # MAYBE MORE FILTERING???
            bin_image = image_bin(one_channel)
            # if Debug:
            #    cv2.imshow("BinaryImage", one_channel)

            frame_img, region_info, history = select_roi(current_frame.copy(), bin_image)

            if Debug:
                cv2.imshow("Return roi", frame_img)
                cv2.setWindowTitle("Return roi", "Return")
            ret_img = process_new_history(frame_img.copy())

            if Debug:
                cv2.imshow("Return processing", ret_img )
                cv2.setWindowTitle("Return processing", "Return processing")
                cv2.waitKey(0)

        for i in range(len(old_history)):
            number = old_history[i].position[0][2]
            if intersect_happened(True, old_history[i]):
                current_sum += number

            if intersect_happened(False, old_history[i]):
                current_sum -= number

        t_sum.append(current_sum)
        print("Video:{0}, sum:{1} ".format(current_video_count, current_sum))

    f = open("out.txt", "w+")
    f.write(student)
    f.write(txt)
    for p in range(len(t_sum)):
        f.write("{:s}\t{:d}\n".format(list_videos[p], t_sum[p]))
    f.close()


def get_all_frames(video):
    path = "../../data/videos/"
    all_frames = []
    vid_cap = cv2.VideoCapture(path + video)
    count = 0
    while True:
        success, img1 = vid_cap.read()
        if not success:
            break
        all_frames.append(img1)
        count += 1
    vid_cap.release()
    return count, all_frames


if __name__ == '__main__':
    model = load_model('model/anmodel.h5')
    videos = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi', 'video-5.avi', 'video-6.avi',
              'video-7.avi', 'video-8.avi', 'video-9.avi']
    calculate(videos)
    test()
