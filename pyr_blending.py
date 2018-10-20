import numpy as np
import cv2


def Gaussian_Pyramid(img, level):
    #achieve the function of cv2.pyrDown
    pyramid = [img]
    for i in range(level):
        blur_img = cv2.GaussianBlur(pyramid[i], (5,5), 0)
        downsample_img = blur_img[::2, ::2]
        pyramid.append(downsample_img)
    
    return pyramid

def Laplacian_Pyramid(img, level):
    pyramid = []
    upper_img = img
    for i in range(level):
        blur_img = cv2.GaussianBlur(upper_img, (5,5), 0)
        lap = upper_img - blur_img
        pyramid.append(lap)
        upper_img = blur_img[::2, ::2]
        if i == level - 1:
            pyramid.append(upper_img)
    
    return pyramid

def blend_pyramid(laplacian_pyr1, laplacian_pyr2, mask_pyr):
    assert len(laplacian_pyr1) == len(laplacian_pyr2), 'pyramid level are not equal'
    assert len(laplacian_pyr1) == len(mask_pyr), 'pyramid level are not equal'
    blended_pyramid = []
    for i in range(len(laplacian_pyr1)):
        blend = mask_pyr[i]*laplacian_pyr1[i] + (1-mask_pyr[i])*laplacian_pyr2[i]
        blended_pyramid.append(blend)
    
    return blended_pyramid

def upsample_img(img):
    shape = list(img.shape)
    shape[0] *= 2
    shape[1] *= 2
    high_res = np.zeros(shape)
    high_res[::2, ::2] = img
    blur = 4*cv2.GaussianBlur(high_res, (5,5), 0)#remeber to multiply 4
    return blur

def recover_laplacian_pyr(lap_pyramid):
    l = len(lap_pyramid)
    out = lap_pyramid[-1]
    for i in reversed(range(0,l-1)):
        upsampled = upsample_img(out)
        if out.shape[0]*2 > lap_pyramid[i].shape[0]:
            upsampled = np.delete(upsampled, -1, axis = 0)
        if out.shape[1]*2 > lap_pyramid[i].shape[1]:
            upsampled = np.delete(upsampled, -1, axis = 1)
        out = upsampled + lap_pyramid[i]
    return out

def main(img1, img2, mask_img, level):
    img1_pyramid = Laplacian_Pyramid(img1, level)
    img2_pyramid = Laplacian_Pyramid(img2, level)
    mask_pyramid = Gaussian_Pyramid(mask_img, level)
    blended_pyramid = blend_pyramid(img1_pyramid, img2_pyramid, mask_pyramid)
    
    return recover_laplacian_pyr(blended_pyramid)

if __name__ == '__main__':
    orange = cv2.imread('orange_test.jpg').astype(np.float64)
    apple = cv2.imread('apple_test.jpg').astype(np.float64)
    mask = np.zeros((orange.shape[0],orange.shape[1],3)).astype(np.float64)
    level = 7
    mask[:, 0:230, :] = 1
    out = np.zeros(orange.shape)
    for i in range(3):
        channel_out = main(apple[:,:,i], orange[:,:,i], mask[:,:,i], level)
        out[:,:,i] = channel_out
    
    cv2.imshow('out', out.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

