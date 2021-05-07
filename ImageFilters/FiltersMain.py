from MorphologicalProcessing import *

original_image = cv2.imread("images/walmart.png")
bw_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
height, width = bw_image.shape

PIXEL_MAX = 255

def invertImage():
    invert_image = np.zeros(shape=bw_image.shape)
    for i in range(0, height):
        for j in range(0, width):
            invert_image[i][j] = PIXEL_MAX - bw_image[i][j]
    return invert_image

cv2.imwrite("ofinal/invert_image.jpg", invertImage())

def logTransform():
    new_image = np.zeros(shape=bw_image.shape)
    c = 255 / (np.log(1 + np.max(bw_image)))
    for i in range(0, height):
        for j in range(0, width):
            pixel = bw_image[i][j]
            n_pixel = c * math.log(pixel + 1)
            new_image[i][j] = n_pixel
    return new_image

cv2.imwrite("ofinal/log_transform.jpg", logTransform())


def inverseLogTransform():
    new_image = np.zeros(shape=bw_image.shape)
    c = 255 / (np.log(1 + np.max(bw_image)))
    for i in range(0, height):
        for j in range(0, width):
            pixel = bw_image[i][j]
            n_pixel = math.pow(math.exp(pixel), 1 / c) - 1
            new_image[i][j] = n_pixel
    return new_image

cv2.imwrite("ofinal/inv_log_tr.jpg", inverseLogTransform())


def powerTransform(gamma):
    new_image = np.zeros(shape=bw_image.shape)
    c = 255 / (np.log(1 + np.max(bw_image)))
    for i in range(0, height):
        for j in range(0, width):
            pixel = bw_image[i][j]
            n_pixel = c * math.pow(pixel, gamma)
            new_image[i][j] = n_pixel
    norm_image = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

cv2.imwrite("ofinal/power_tr.jpg", powerTransform(0.8))


def constrastStretching(image = bw_image, a=0, b=255, c=40, d=170):
    # c = 40  # np.min(bw_image)
    # d = 170  # np.max(bw_image)
    new_image = np.zeros(shape=image.shape)
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i][j]
            n_pixel = ((pixel - c) * ((b - a) / (d - c))) + a
            new_image[i][j] = n_pixel
    norm = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm


cv2.imwrite("ofinal/contrast_str.jpg", constrastStretching())


def bitPlaneSlicing():
    new_image = np.zeros(shape=bw_image.shape)
    for i in range(0, height):
        for j in range(0, width):
            pixel = bw_image[i][j]
            n_pixel = format(pixel, "08b")[0]
            new_image[i][j] = int(n_pixel) * 255
    return new_image

cv2.imwrite("ofinal/bit_plane.jpg", bitPlaneSlicing())

def getMedianFilterValue(i, j, kernalSize):
    ps = []
    for x in range(-kernalSize, kernalSize + 1):
        for y in range(-kernalSize, kernalSize + 1):
            pixel = bw_image[i + x][j + y]
            ps.append(pixel)
    ps.sort()
    return ps[int(kernalSize * kernalSize / 2)]



def medianFilter():
    new_image = np.zeros(shape=bw_image.shape)
    kernelSize = 3
    intKS = int(kernelSize / 2)
    for i in range(intKS, height - intKS):
        for j in range(intKS, width - intKS):
            pixel = getMedianFilterValue(i, j, intKS)
            new_image[i][j] = pixel
    return new_image

cv2.imwrite("ofinal/median_filter.jpg", medianFilter())


def getAverageFilterKernel(i, j, kernalSize):
    kernel = []
    for x in range(-kernalSize, kernalSize + 1):
        row = []
        for y in range(-kernalSize, kernalSize + 1):
            row.append(bw_image[i + x][j + y])
        kernel.append(row)
    return kernel



def getAverageFilterPixelValue(kernel, kernalSize):
    avg = 0
    for x in range(0, kernalSize):
        for y in range(0, kernalSize):
            temp = int(kernel[x][y])
            avg = temp + avg

    return int(avg / math.pow(kernalSize, 2))


def applyAvgFilter():
    new_image = np.zeros(shape=bw_image.shape)
    kernelSize = 3
    intKS = int(kernelSize / 2)
    for i in range(intKS, height - intKS):
        for j in range(intKS, width - intKS):
            kernel = getAverageFilterKernel(i, j, intKS)
            new_image[i][j] = getAverageFilterPixelValue(kernel, kernalSize=kernelSize)
    return new_image

cv2.imwrite("ofinal/avg.jpg", applyAvgFilter())


def getGaussianFilterPixel(i, j, kernel, kernelSize):
    value = 0
    for x in range(0, kernelSize):
        for y in range(0, kernelSize):
            value = value + int(bw_image[i - 1 + x][j - 1 + y]) * kernel[x][y]
    return value / 16



def applyGuassinFilter(image=bw_image):
    new_image = np.zeros(shape=image.shape)
    kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            new_image[i][j] = getGaussianFilterPixel(i, j, kernel, 3)
    norm_image = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

cv2.imwrite("ofinal/gaussian.jpg", applyGuassinFilter())

def getLaplicanFilterPixel(i, j, kernel, kernelSize):
    value = 0
    for x in range(0, kernelSize):
        for y in range(0, kernelSize):
            value = value + int(bw_image[i - 1 + x][j - 1 + y]) * kernel[x][y]
    return value



def applyLaplacian():
    new_image = np.zeros(shape=bw_image.shape)
    kernel = [[0,-1,0], [-1,4,-1], [0,-1,0]]
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            new_image[i][j] = getLaplicanFilterPixel(i, j, kernel, 3)
    norm_image = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

cv2.imwrite("ofinal/laplacian.jpg", applyLaplacian())


def applyUnsharpMasking(image = bw_image):
    g = np.add(image , applyLaplacian())
    norm_image = cv2.normalize(g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

cv2.imwrite("ofinal/unsharp_masking.jpg", applyUnsharpMasking())



