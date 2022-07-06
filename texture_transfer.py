import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.color.colorconv import gray2rgb, rgba2rgb
from skimage.filters import gaussian
from skimage import io, util, filters
import heapq
import time
from numba import jit
import cv2
import gradio as gr

@jit
def randomPatch(texture, patchLength): # generates random values that select part of the texture and returns it as a patch
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength) # set i var (y coord) as a random int
    j = np.random.randint(w - patchLength) # set j var (x coord) as a random int

    return texture[i:i+patchLength, j:j+patchLength] # returns a random patch of the texture within range of patchLength


@jit
def L2OverlapDiff(patch, patchLength, overlap, res, y, x): # calculates patch overlap
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += np.sum(up**2)
 
    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error
 

@jit
def randomBestPatch(texture, patchLength, overlap, res, y, x): 
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength)) # errors var set as the size for stitched block boundary set by L2OverlapDiff

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+patchLength, j:j+patchLength]
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x) # set e as an error of a certain boundary between two blocks
            errors[i, j] = e # populate errors var with calculated error for overlapping boundaries

    i, j = np.unravel_index(np.argmin(errors), errors.shape) # sets i and j with coordinates based on the minimum error values in errors
    return texture[i:i+patchLength, j:j+patchLength]



@jit
def minCutPath(errors): #determines minimum error stitching path 
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


@jit
def minCutPath2(errors):
    # dynamic programming, unused
    errors = np.pad(errors, [(0, 0), (1, 1)], 
                    mode='constant', 
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)    

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        # optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)
    
    paths -= 1
    
    minCutPath = [np.argmin(cumError)]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])
    
    return map(lambda x: x - 1, reversed(minCutPath))


@jit
def minCutPatch(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


@jit
def quilt(texture, patchLength, numPatches, mode="cut", sequence=False): 
    texture = util.img_as_float(texture)

    overlap = patchLength // 6 
    numPatchesHigh, numPatchesWide = numPatches

    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))

    for i in range(numPatchesHigh): #iterates through height of the entire img to be generated  
        for j in range(numPatchesWide):  # ^^ width 
            y = i * (patchLength - overlap)  #locates y coords of each patch
            x = j * (patchLength - overlap)  #locates x coords of each patch

            if i == 0 and j == 0 or mode == "random":   # conditionals to check which approach to use based on specified mode
                patch = randomPatch(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch # assigns patch generated to res 2D array

            if sequence:
                io.imshow(res)
                io.show()
      
    return res


@jit
def quiltSize(texture, patchLength, shape, mode="cut"):
    overlap = patchLength // 6
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), mode)

    return res[:h, :w]

# texture = io.imread("test.png")
# io.imshow(texture)
# io.show()

# io.imshow(quilt(texture, 25, (6, 6), "random"))
# io.show()

# io.imshow(quilt(texture, 25, (6, 6), "best"))
# io.show()

# io.imshow(quilt(texture, 20, (6, 6), "cut"))
# io.show()

# io.imshow(quilt(texture, 20, (3, 3), "cut", True))
# io.show()

@jit
def bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i+curPatchHeight, j:j+curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = np.sum(e**2)

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+curPatchHeight, j:j+curPatchWidth]

@jit
def bestCorrOverlapPatch(texture, corrTexture, patchLength, overlap, 
                         corrTarget, res, y, x, alpha=0.1, level=0):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+di, j:j+dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = np.sum(l2error)

            corrTexturePatch = corrTexture[i:i+di, j:j+dj]
            corrError = np.sum((corrTexturePatch - corrTargetPatch)**2)

            prevError = 0
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y+overlap:y+patchLength, x+overlap:x+patchLength]
                prevError = np.sum(prevError**2)
            
            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+di, j:j+dj]



@jit
def transfer(texture, target, patchLength, mode="cut", 
             alpha=0.1, level=0, prior=None, blur=False):

    # transform texture and target images to grayscale 
    corrTexture = rgb2gray(texture) 
    corrTarget  = rgb2gray(target)

    # sobelX = cv2.Sobel(target, cv2.CV_64F, 1, 0)
    # sobelY = cv2.Sobel(target, cv2.CV_64F, 0, 1)

    # sobelX = np.uint8(np.absolute(sobelX))
    # sobelY = np.uint8(np.absolute(sobelY))

    #sobelX = filters.sobel_h(corrTarget)
    #sobelY = filters.sobel_v(corrTarget)

    corrTarget = sobelX * sobelY

    if blur:
        corrTexture = gaussian(corrTexture, sigma=3)
        corrTarget  = gaussian(corrTarget,  sigma=3)

    io.imshow(corrTexture)
    io.show()
    io.imshow(corrTarget)
    io.show()

    # remove alpha channel
    texture = util.img_as_float(texture)[:,:,:3]
    target = util.img_as_float(target)[:,:,:3]

    h, w, _ = target.shape #get target image height and width
    overlap = patchLength // 6 

    # divide target image's h and w dimensions for number of patches 
    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    
    if level == 0:
        res = np.zeros_like(target)
    else:
        res = prior

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap) # i * (20 - 3)  
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x, 
                                             alpha, level)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch
      
    return res

@jit
def transferIter(texture, target, patchLength, n): # iterates transfer n times, manipulating transfer() parameters
    res = transfer(texture, target, patchLength) 
    for i in range(1, n):
        alpha = 0.1 + 0.8 * i / (n - 1) # switches alpha values every iteration (between 1 and 2)
        patchLength = patchLength * 2**i // 3**i # increases patchLength after each iteration
        print((alpha, patchLength)) 
        res = transfer(texture, target, patchLength, 
                       alpha=alpha, level=i, prior=res) # apply new values to new transfer() call
    
    return res

@jit
def Loss_function(original, syn):
  height, width, depth = original.shape
  for i in range(height):
      loss3 += np.sqrt(np.sum(np.square(original[i][:,0:3]/np.max(original) - syn[i]/np.max(syn))))

def style_transfer(texture, target):
    res2 = transferIter(texture, target, 20, 2)
    return ("Please come back after a few minutes, the image should appear below",res2)

@jit
def main():
    interface = gr.Interface(fn=style_transfer, inputs=['image', 'image'], outputs=["text","image"])
    interface.launch()

if __name__ == "__main__":
    main()