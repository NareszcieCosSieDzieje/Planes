
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from statistics import median,mean
import random
import os

kernel = np.ones((5,5),np.uint8)
fig, ax_arr = plt.subplots(6, 3, squeeze=True)
image_list = list()

def process_images(showImgs = True):
    for h in range(21):
        if h in [6,18,19]:
            continue

        path = 'Resources/PRE/samolot'
        save_path = 'Resources/POST/samolot'
        add_string = str()
        if h < 10:
              add_string = ("0" + str(h) + ".jpg")
        else:
            add_string = (str(h) + ".jpg")
        path += add_string
        save_path += add_string
        save_path += '.jpg'

        plane_image = cv2.imread(path, 0)
        color_image = cv2.imread(path, 1)

        blur = cv2.GaussianBlur(plane_image, (5, 5), 0)
        blur = cv2.fastNlMeansDenoising(blur, None, 9, 13)
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        threshold = (255 - threshold)
        img = cv2.dilate(threshold, kernel, iterations=1)

        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        img_out = img | im_floodfill_inv

        img = cv2.erode(img_out, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = list()  # list to hold all areas
        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        mean_area = mean(areas)
        max_area = max(areas)
        cnt = list()

        #print("max: ", max_area)
        #print("sred: ", mean_area)
        #print("max/sr: ", max_area/mean_area)
        #median_area = median(areas)
        #print("mediana: ", median_area)

        for contour in contours:
            ar = cv2.contourArea(contour)
            if (max_area > mean_area*50):
                if (ar == max_area):
                    cnt.append(contour)
            elif (ar >= 0.65*mean_area):
                cnt.append(contour)

        #cv2.drawContours(color_image, cnt, -1, (0, 0, 255), 2)
        color_list = list()
        for contour in cnt:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            while(True):
                b = random.randint(0, 255)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                px = [b,g,r]
                if px not in color_list:
                    color_list.append(px)
                    break
            cv2.drawContours(color_image, [contour], -1, color_list[-1], 3)
            cv2.circle(color_image, (cX, cY), 5, (255, 255, 255), -1)
        image_list.append(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if os.path.exists(save_path):
            os.remove(save_path)
        matplotlib.image.imsave(save_path, image_list[-1])
        if showImgs:
            cv2.namedWindow(path, cv2.WINDOW_NORMAL)
            cv2.imshow(path, color_image)
            cv2.waitKey(0)
    if showImgs:
        cv2.destroyAllWindows()

def make_plot():
    iter=0
    for col in range(6):
        for row in range(3):
            ax_arr[col, row].imshow(image_list[iter])
            ax_arr[col, row].axis('off')
            iter+=1
    plt.tight_layout()
    fig.subplots_adjust(hspace=None, wspace=None)
    plt.subplots_adjust(wspace=None, hspace=None)
    if os.path.exists("Resources/calosc.pdf"):
        os.remove("Resources/calosc.pdf")
    plt.savefig('Resources/calosc.pdf')
    plt.show()

def main():
    process_images(False)
    #make_plot()
    #for i in range(18):
     #   plt.imshow(image_list[i])
    #    plt.show()
    path = "C:/Users/paulc/Desktop/Semestr_5/KCK/CW4/images/Resources/POST"
    path = os.path.realpath(path)
    os.startfile(path)

if __name__ == "__main__":
    main()