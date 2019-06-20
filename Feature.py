from typing import NamedTuple
from time import time
from imageio import imwrite
from numpy import array, zeros, sqrt, indices, intp, rint, gradient, where, all as np_all, average
from scipy.ndimage import zoom, gaussian_filter, spline_filter, generate_binary_structure
from scipy.ndimage.measurements import label, find_objects

''' Literature: Lindeberg 2015, "Image Matching Using Generalized Scale-Space Interest Points" '''

class Featurepoint(NamedTuple):
    x: float
    y: float
    scale: float
    category: str

def normalizeImageArray(imageArray):
    ''' Normalization to intervall [0,1] '''
    imageArray/=255
    minval=imageArray.min()
    maxval=imageArray.max()
    if minval>0 or maxval<1:
        imageArray=(imageArray-minval)/(maxval-minval)
    return imageArray

def calcImagePyramid(imageArray, resolutionMax, scalerangeFactor, scalesCount):
    ''' Parameter:
        - resolutionMax :: positive Float [MPix]; resolution of the image with highest resolution in the pyramid
        - scalerangeFactor :: positive Float; factor between smallest and largest scale (zoom-level)
        - scalesCount :: postive Int; Number of scales '''
    resolutionFull=imageArray.size/10**6 # full resolution [MPix]
    assert(0<=resolutionMax<=resolutionFull)
    # zoomfactor of the image with highest/lowest resolution in the pyramid:
    zoomMax=sqrt(resolutionMax/resolutionFull)
    zoomMin=zoomMax/scalerangeFactor

    start=time()
    #The first scale-image is stored with the by zoomMax parameterized resulution. Further normalize the imageArray, so the hessianThreshold becomes independent of image brightness:
    imageArrayZoomMax=normalizeImageArray(zoom(imageArray,zoom=zoomMax,prefilter=True))
    imagePyramid=[imageArrayZoomMax]
    scales=[(imagePyramid[0].shape[1]/imageArray.shape[1],
             imagePyramid[0].shape[0]/imageArray.shape[0])]
    #TODO: prefilter=True (spline filter) yields sometimes artefacts, but without the images get gradually to blurry. It is also necessary to examine the extent to which the images of the following scaling levels must be filtered.
    deltaZoom=(zoomMin/zoomMax)**(1/(scalesCount-1)) if scalesCount>1 else None
    for idxScale in range(1,scalesCount):
        scaleImage=zoom(imagePyramid[idxScale-1],zoom=deltaZoom,prefilter=True)
        scale = (scaleImage.shape[1]/imageArray.shape[1] ,scaleImage.shape[0]/imageArray.shape[0])
        imagePyramid.append(scaleImage)
        scales.append(scale)
    print('Time for calculate imagePyramid: '+str(time()-start))
    return (imagePyramid, scales)

def calcGradients(imageArray):
    ''' calculates gradients in x- and y-direction '''
    Ix=gradient(imageArray, axis=1)
    Iy=gradient(imageArray, axis=0)
    gradients=array([Ix,Iy])
    return gradients

def calcHessianPyramid(imagePyramid, scalesCount):
    ''' calculates entrys of hessian-matrix and its determinant for every scale '''
    hessianPyramid=[]
    start=time()
    for idxScale in range(scalesCount):
        gradients=calcGradients(imagePyramid[idxScale])
        Ixx=gradient(gradients[0], axis=1)
        Ixy=gradient(gradients[0], axis=0)
        Iyy=gradient(gradients[1], axis=0)
        detH=Ixx*Iyy-Ixy**2
        hessianPyramid.append(array([Ixx,Ixy,Iyy,detH]))
    print('Time for calculate hessians: '+str(time()-start))
    return hessianPyramid

def detectFeaturepoints(
        imageArray,
        resolutionMax,
        scalerangeFactor,
        scalesCount,
        hessianThreshold,
        minFeatureScales):
    ''' Parameter:
        - resolutionMax :: positive Float [MPix]; resolution of the image with highest resolution in the pyramid
        - scalerangeFactor :: positive Float; factor between smallest and largest scale (zoom-level)
        - scalesCount :: postive Int; Number of scales
        - minFeatureScales :: positive Int; Minimum number of scales in which a feature occurs
        - hessianThreshold :: positive Float; Threshold for detection of critical points '''
    start=time()
    (imagePyramid, scales) = calcImagePyramid(imageArray, resolutionMax, scalerangeFactor, scalesCount)
    hessianPyramid = calcHessianPyramid(imagePyramid, scalesCount)
    extrema={'minima':[], 'maxima':[], 'saddle':[]}
    for idxScale in range(scalesCount):
        #sufficient criterion for extrema (Hessematrix positive- or negative-definite). These equivalences apply here: H positive-definite: Ixx>0 and det(H)>0  |  H negative-definite: Ixx<0 and det(H)>0 (see literature: Lindeberg 2015, "Image Matching Using Generalized Scale-Space Interest Points", p.9)
        # TODO: check if an additional threshold for Ixx,Iyy or Ixy brings an improvement
        Ixx=hessianPyramid[idxScale][0]
        detH=hessianPyramid[idxScale][3]
        indicesMinima=where(np_all([detH > hessianThreshold, Ixx > 0], axis=0))
        indicesMaxima=where(np_all([detH > hessianThreshold, Ixx < 0], axis=0))
        indicesSaddle=where(detH < -hessianThreshold)
        extrema['minima'].append(array([indicesMinima[1],indicesMinima[0]]).T)
        extrema['maxima'].append(array([indicesMaxima[1],indicesMaxima[0]]).T)
        extrema['saddle'].append(array([indicesSaddle[1],indicesSaddle[0]]).T)

    print('Time for find extrema: '+str(time()-start))


    start=time()
    # Averaging of the extrema in scale-space over their spans:
    # TODO: Improve averaging of extrema (subpixel interpolation)
    featurepoints=[]
    for category in extrema:
        responseArray=zeros(shape=(scalesCount,)+imagePyramid[0].shape, dtype=float)
        subpixelArray=zeros(shape=(2,scalesCount)+imagePyramid[0].shape, dtype=float)
        labelStructure=generate_binary_structure(3,3)
        # TODO: Does this labelStructure also identifies diagonally adjacent pixels as connected?
        for idxScale in range(scalesCount):
            #pixelcoordinates of the extrema:
            extremaPixel=\
                rint(array([extrema[category][idxScale][:,0],
                            extrema[category][idxScale][:,1]]).T).astype(intp)
            #Transform the pixel-coordinates all into the coordinate-system of the first scale-level and round them to the next integer to get the index in scale-space:
            extremaPixelScaleSpace = \
                array([extrema[category][idxScale][:,0] * scales[0][0] / scales[idxScale][0],
                       extrema[category][idxScale][:,1] * scales[0][1] / scales[idxScale][1],
                       [idxScale] * len(extrema[category][idxScale])]).T #Order: x,y,scale
            extremaPixelScaleSpaceRound=rint(extremaPixelScaleSpace).astype(intp)

            #The feature-response (in this case detH) is set in the responseArray at the rounded indexpositions of the extrema:
            responseArray[
                idxScale,
                extremaPixelScaleSpaceRound[:,1],
                extremaPixelScaleSpaceRound[:,0]] = \
                    hessianPyramid[idxScale][3,extremaPixel[:,1],extremaPixel[:,0]]
            #The unrounded subpixel-coordinates in scaleSpace are stored in the subpixelArray at the rounded indexpositions of the extrema:
            subpixelArray[
                0, #x-direction
                idxScale,
                extremaPixelScaleSpaceRound[:,1],
                extremaPixelScaleSpaceRound[:,0]] = \
                    extremaPixelScaleSpace[:,0]
            subpixelArray[
                1, #y-direction
                idxScale,
                extremaPixelScaleSpaceRound[:,1],
                extremaPixelScaleSpaceRound[:,0]] = \
                    extremaPixelScaleSpace[:,1]

        #Connected Environments in scaleSpace belong to the same feature. This environments are averaged to calculate the featurepositions with subpixel accuracy:
        scaleSpaceExtremaLabeled, featureCount=label(responseArray, labelStructure)
        featureSlices=find_objects(scaleSpaceExtremaLabeled)
        for i in range(featureCount):
            # For each feature, its span is selected from the response array:
            responseArrayFeature=responseArray[featureSlices[i]]
            # Pixel coordinates of the feature environment:
            indicesScale, indices_y, indices_x =indices(responseArrayFeature.shape)
            scale0=featureSlices[i][0].start
            y0    =featureSlices[i][1].start
            x0    =featureSlices[i][2].start
            indicesScale+=scale0
            indices_y+=y0
            indices_x+=x0
            # To average the feature coordinates, do not take the rounded pixels, but the unrounded coordinates from the subpixelArray on the slice of the feature:
            # TODO: Maybe also save scale in subpixelArray instead of using indicesScale?
            coordsX=subpixelArray[0][featureSlices[i]].ravel()
            coordsY=subpixelArray[1][featureSlices[i]].ravel()
            featureNeighborhoodCoordinatesSubpixel=\
                array([indicesScale.ravel(),coordsY.ravel(),coordsX.ravel()]).T
            # Dismiss features which featureScaleRange spans not the minimum number of scales. Such features are to 'weak':
            idxScaleMin=featureNeighborhoodCoordinatesSubpixel[0,0]
            idxScaleMax=featureNeighborhoodCoordinatesSubpixel[-1,0]
            featureScaleRange=idxScaleMax-idxScaleMin+1
            if featureScaleRange<minFeatureScales:
                continue
            # The weights of the individual points in the feature environment are determined by the feature response detH.
            neighborhoodWeights=responseArrayFeature.ravel()
            # TODO: Examine if absolute values ​​of detH is the better weighting.
            # neighborhoodWeights=np_abs(responseArrayFeature).ravel()

            # The coordinates of the feature are finally averaged over the neighborhood:
            coordinatesFeature=average(
                featureNeighborhoodCoordinatesSubpixel, axis=0, weights=neighborhoodWeights)
            scaleFeature=coordinatesFeature[0]
            # The feature coordinates are specified in coordinates of the full-resolution image:
            featurepoint = Featurepoint(
                x=coordinatesFeature[2]/scales[0][0],
                y=coordinatesFeature[1]/scales[0][1],
                scale=scaleFeature,
                category=category)
            featurepoints.append(featurepoint)
    print('Time for detection '+str(len(featurepoints))+' of featurepoints: '+str(time()-start))
    return featurepoints


