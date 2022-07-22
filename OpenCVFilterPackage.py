import math
import numpy as np
import cv2
from scipy.ndimage import rotate, shift

def flip_img(img, flip_axis):
        """flip img along specified axis(x or y)"""
        return np.flip(img, flip_axis)

def shift_img(img, shift_range, shift_axis):
        """shift img by specified range along specified axis(x or y)"""
        shift_lst = [0] * img.ndim
        shift_lst[shift_axis] = math.floor(shift_range * img.shape[shift_axis])
        return shift(img, shift=shift_lst, cval=0)

def resize_img(img, scale_percent):
        """Resize image using """
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        zoomed = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return zoomed

def rotate_img(img, rotate_axis,rotate_angle):
        """rotate img by specified range along specified axis(x or y)"""
        return rotate(img, axes=rotate_axis, angle=rotate_angle, cval=0.0, reshape=False)

def sharpen_img_A(img):
        #Edge enhance
        kernel = np.array([[-1,-1,-1,-1,-1],
                    [-1,2,2,2,-1],
                    [-1,2,8,2,-1],
                    [-2,2,2,2,-1],
                    [-1,-1,-1,-1,-1]])/8.0
        result=cv2.filter2D(img,-1,kernel)
        return result

def adjust_gamma_img(img, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

def sharpen_img_B(img):
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        result=cv2.filter2D(img,-1,kernel)
        return result

def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def saturation(image, saturation):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if saturation >= 0:
        lim = 255 - saturation
        s[s > lim] = 255
        s[s <= lim] += saturation
    else:
        saturation = abs(saturation)
        lim = saturation
        s[s < lim] = 0
        s[s >= lim] -= saturation

    saturated_image = cv2.merge((h, s, v))
    return cv2.cvtColor(saturated_image, cv2.COLOR_HSV2RGB)

def contrast(image, contrast):
    temp_img = np.int16(image)
    temp_img = temp_img * (contrast/127+1) - contrast
    temp_img = np.clip(temp_img, 0, 255)
    return np.uint8(temp_img)

namefile="test_img.png"


img = cv2.imread(namefile)


flip = flip_img(img, 0)
imgshift = shift_img(img, 0.05, 1)
resize = resize_img(img,  90)
rotateimg = rotate_img(img, (0,1), 90)
sharpenA = sharpen_img_A(img)
sharpenB = sharpen_img_B(img)
gammaadjusted = adjust_gamma_img(img, 0.5)
unsharpmask = unsharp_mask(img)
imgsaturation = saturation(img, 40)
imgcontrast = contrast(img, 80)

cv2.imshow("IMG",img)
cv2.imshow("contrast",imgcontrast)
cv2.imshow("saturation",imgsaturation)
cv2.imshow("flip",flip)
cv2.imshow("shift",imgshift)
cv2.imshow("resize",resize)
cv2.imshow("rotate",rotateimg)
cv2.imshow("sharpenA",sharpenA)
cv2.imshow("sharpenB",sharpenB)
cv2.imshow("gammaadjusted",gammaadjusted)
cv2.imshow("unsharpmask",unsharpmask)


cv2.waitKey(0)
