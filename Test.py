import os
from imageio import imread
import matplotlib.pyplot as plt

from Feature import detectFeaturepoints

def testDetectFeaturepoints():
    imageArray = imread(os.path.dirname(__file__)+'/img/ruin.JPG', as_gray=True)
    resolutionMax = 1
    scalerangeFactor = 3
    scalesCount = 8
    hessianThreshold = 0.03
    minFeatureScales = 3
    featurepoints = detectFeaturepoints(
                        imageArray,
                        resolutionMax,
                        scalerangeFactor,
                        scalesCount,
                        hessianThreshold,
                        minFeatureScales)
    featurepointsX = [f.x for f in featurepoints]
    featurepointsY = [f.y for f in featurepoints]

    plt.imshow(imageArray, cmap = plt.get_cmap('gray'))
    plt.scatter(featurepointsX,featurepointsY,s=1,c="red")
    plt.savefig("ruinFeatures.jpg")

if __name__ == "__main__":
    testDetectFeaturepoints()