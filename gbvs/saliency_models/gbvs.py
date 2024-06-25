import time
import cv2
import numpy as np
import scipy.io
import sklearn.preprocessing
import math
import numpy as np
import numpy.matlib

def markovChain_solve(mat, tolerance):
    w,h = mat.shape
    diff = 1
    v = np.divide(np.ones((w, 1), dtype=np.float32), w)
    oldv = v
    oldoldv = v

    while diff > tolerance :
        oldv = v
        oldoldv = oldv
        v = np.dot(mat,v)
        diff = np.linalg.norm(oldv - v, ord=2)
        s = sum(v)
        if s>=0 and s< np.inf:
            continue
        else:
            v = oldoldv
            break

    v = np.divide(v, sum(v))

    return v

def getGaborKernel(gaborparams, angle, phase):
    gp = gaborparams
    major_sd = gp['stddev']
    minor_sd = major_sd * gp['elongation']
    max_sd = max(major_sd, minor_sd)

    sz = gp['filterSize']
    if sz == -1:
        sz = math.ceil(max_sd * math.sqrt(10))
    else:
        sz = math.floor(sz / 2)

    psi = np.pi / 180 * phase
    rtDeg = np.pi / 180 * angle

    omega = 2 * np.pi / gp['filterPeriod']
    co = math.cos(rtDeg)
    si = -math.sin(rtDeg)
    major_sigq = 2 * pow(major_sd, 2)
    minor_sigq = 2 * pow(minor_sd, 2)

    vec = range(-int(sz), int(sz) + 1)
    vlen = len(vec)
    vco = [i * co for i in vec]
    vsi = [i * si for i in vec]

    # major = np.matlib.repmat(np.asarray(vco).transpose(), 1, vlen) + np.matlib.repmat(vsi, vlen, 1)
    a = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vsi, vlen, 1)
    major = a + b
    major2 = np.power(major, 2)

    # minor = np.matlib.repmat(np.asarray(vsi).transpose(), 1, vlen) - np.matlib.repmat(vco, vlen, 1)
    a = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vco, vlen, 1)
    minor = a + b
    minor2 = np.power(minor, 2)

    a = np.cos(omega * major + psi)
    b = np.exp(-major2 / major_sigq - minor2 / minor_sigq)
    # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
    result = np.multiply(a, b)

    filter1 = np.subtract(result, np.mean(result.reshape(-1)))
    filter1 = np.divide(filter1, np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
    return filter1


def getGaborKernels(gaborparams, thetas):
    gaborKernels = {}
    for th in thetas:
        gaborKernels[th] = {}
        gaborKernels[th]['0'] = getGaborKernel(gaborparams, th, 0)
        gaborKernels[th]['90'] = getGaborKernel(gaborparams, th, 90)

    return gaborKernels

def orientationFeatureMaps_compute(L, gaborparams, thetas):
    # L = Intensity Map
    # L = np.maximum(np.maximum(r, g), b)

    kernels = getGaborKernels(gaborparams, thetas)
    featMaps = []
    for th in thetas:
        kernel_0  = kernels[th]['0']
        kernel_90 = kernels[th]['90']
        o1 = cv2.filter2D(L, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
        o2 = cv2.filter2D(L, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)
        o = np.add(abs(o1), abs(o2))
        featMaps.append(o)

    return featMaps

def loadGraphDistanceMatrixFor28x32():
    #预处理
    f = scipy.io.loadmat("../gbvs/saliency_models/28__32__m__2.mat")
    #推理
    # f = scipy.io.loadmat("gbvs/saliency_models/28__32__m__2.mat")
    distanceMat = np.array(f['grframe'])[0][0][0]
    lx = np.array(f['grframe'])[0][0][1]
    dim = np.array(f['grframe'])[0][0][2]
    return [distanceMat, lx, dim]

def calculate(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major

    state_transition_matrix = Fab * np.abs(
        (np.zeros((distanceMat.shape[0], distanceMat.shape[1])) + map_linear).T - map_linear
    ).T

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain_solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped

def normalize(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major
    # calculating STM : w = d*Fab
    state_transition_matrix = (Fab.T * np.abs(map_linear)).T

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    # print(np.isnan(state_transition_matrix).any())
    state_transition_matrix[np.isnan(state_transition_matrix)] = 1
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain_solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped


def compute(r, g, b, L):
    # Input is the r, g, b channels of the image

    #CBY Feature Map
    min_rg = np.minimum(r, g)
    b_min_rg = np.abs(np.subtract(b, min_rg))
    CBY = np.divide(b_min_rg, L, out=np.zeros_like(L), where=L != 0)

    #CRG Feature Map
    r_g = np.abs(np.subtract(r,g))
    CRG = np.divide(r_g, L, out=np.zeros_like(L), where=L != 0)

    featMaps = {}
    featMaps['CBY'] = CBY
    featMaps['CRG'] = CRG
    featMaps['L'] = L
    return featMaps

def calculateFeatureMaps(r, g, b, L, params):
    colorMaps = compute(r, g, b, L)
    orientationMaps = orientationFeatureMaps_compute(L, params['gaborparams'], params['thetas'])
    allFeatureMaps = {
        0: colorMaps['CBY'],
        1: colorMaps['CRG'],
        2: colorMaps['L'],
        3: orientationMaps
    }
    return allFeatureMaps

def getPyramids(image, max_level):
    imagePyr = [cv2.pyrDown(image)]
    for i in range(1, max_level):
        # imagePyr.append(cv2.resize(p, (32, 28), interpolation=cv2.INTER_CUBIC))
        imagePyr.append(cv2.pyrDown(imagePyr[i-1]))
    return imagePyr[1:]

def run(image, params):
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    L = np.maximum(np.maximum(r, g), b)

    b_pyr = getPyramids(b, params['max_level'])
    g_pyr = getPyramids(g, params['max_level'])
    r_pyr = getPyramids(r, params['max_level'])
    L_pyr = getPyramids(L, params['max_level'])

    featMaps = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    # calculating feature maps

    for i in range(0, len(b_pyr)):
        p_r = r_pyr[i]
        p_g = g_pyr[i]
        p_b = b_pyr[i]
        p_L = L_pyr[i]

        maps = calculateFeatureMaps(p_r, p_g, p_b, p_L, params)
        # we calculate feature maps and then resize
        for i in range(0,3):
            resized_m = cv2.resize(maps[i], (32, 28), interpolation=cv2.INTER_CUBIC)
            featMaps[i].append(resized_m)

        for m in maps[3]:
            resized_m = cv2.resize(m, (32, 28), interpolation=cv2.INTER_CUBIC)
            featMaps[3].append(resized_m)
        # featMaps[0].append(maps[0])
        # featMaps[1].append(maps[1])
        # featMaps[2].append(maps[2])

    # calculating activation maps

    activationMaps = []
    activation_sigma = params['sigma_frac_act']*np.mean([32, 28]) # the shape of map

    for i in range(0,4):
        for map in featMaps[i]:
            activationMaps.append(calculate(map, activation_sigma))


    # normalizing activation maps

    normalisedActivationMaps = []
    normalisation_sigma = params['sigma_frac_norm']*np.mean([32, 28])

    for map in activationMaps:
        normalisedActivationMaps.append(normalize(map, normalisation_sigma))


    # combine normalised maps

    mastermap = normalisedActivationMaps[0]
    for i in range(1, len(normalisedActivationMaps)):
        mastermap = np.add(normalisedActivationMaps[i], mastermap)


    # post process

    gray = cv2.normalize(mastermap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # blurred = cv2.GaussianBlur(gray,(4,4), 4)
    # gray2 = cv2.normalize(blurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mastermap_res = cv2.resize(gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    return mastermap_res

def setupParams():
    gaborparams = {
        'stddev': 2,
        'elongation': 2,
        'filterSize': -1,
        'filterPeriod': np.pi
    }

    params = {
        'gaborparams': gaborparams,
        'sigma_frac_act': 0.15,
        'sigma_frac_norm': 0.06,
        'max_level': 4,
        'thetas': [0, 45, 90, 135]
    }

    return params


def compute_saliency(input_image):
    if type(input_image) is str:
        input_image = cv2.imread(input_image)

    params = setupParams()
    return run(image=input_image / 255.0, params=params) * 255.0


if __name__ == '__main__':
    image = "E:\code\ResShift-unet\S10110_origin_0.jpg"
    # image = r"E:\code\ResShift-unet\temp_img\gbvs-master\images\1.jpg"

    # 读取图像
    image = cv2.imread(image)
    print(image.shape)
    # 将图像转换为 NumPy 数组
    image_np = np.array(image, dtype=np.float32)

    # 将 NaN 值替换为 0
    image_np[np.isnan(image_np)] = 1
    nan_mask = np.isnan(image_np)
    if np.any(nan_mask):
        print("Image contains NaN values.")
        # 输出包含 NaN 值的像素的索引
        nan_indices = np.argwhere(nan_mask)
        print("Indices of NaN values:")
        print(nan_indices)
    else:
        print("Image does not contain NaN values.")

    # 将 NumPy 数组转换回 uint8 类型，以便保存或显示
    image_fixed = image_np.astype(np.uint8)

    saliency_map_gbvs = compute_saliency(image_fixed)
    _saliency_map_gbvs = np.repeat(saliency_map_gbvs[:, :, np.newaxis], 3, axis=2)


    _candy = cv2.Canny(image,0,15)
    r,g,b = cv2.split(image)
    add = (r+g+b)/3
    saliency_map_gbvs2 = saliency_map_gbvs + add
    saliency_map_gbvs2 = np.repeat(saliency_map_gbvs2[:, :, np.newaxis], 3, axis=2)
    # saliency_map_gbvs3 = np.stack(saliency_map_gbvs, _candy)


    print(_saliency_map_gbvs.shape,saliency_map_gbvs2.shape)

    # saliency_map_ikn = ittikochneibur.compute_saliency(image_fixed)

    oname = "gbvs.jpg"
    cv2.imwrite(oname, _saliency_map_gbvs)
    cv2.imwrite("gbvs2.jpg", saliency_map_gbvs2)

    # cv2.imwrite("gbvs3.jpg", saliency_map_gbvs3)
