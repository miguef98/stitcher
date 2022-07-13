import numpy as np
import matplotlib.pyplot as plt
import cv2
from stitcher import Imagen


class StitcherGD:
    def stitch( self, images ,ratio=0.75, reprojThresh=4 ):
        if len(images) < 2:
            raise Exception("Se necesitan al menos dos imagenes")
        
        # las paso a mi clase Image
        images = [ Imagen(image, descriptor="SIFT") for image in images ]

        gs = []
        Ts = [np.eye(3)]
        
        for i in range(len(images) - 1):
            (g_i, T_i, inliners) = self.matchImages(images[i], images[i+1], ratio, reprojThresh)
            gs.append([g_i[i] for i in range(len(inliners)) if inliners[i] == 1])
            Ts.append(T_i)

        Ts = np.array(Ts)

        errores = []

        delta = 0.0000001
        error = np.finfo(np.float32).max
        epsilon = np.inf
        #while np.abs(epsilon) > 0.5:
        for _ in range(3):
            nuevo_error = self.error(images, Ts, gs)
            errores.append(nuevo_error)
            gradientes = self.gradient(images, Ts, gs)
            Ts -= delta * gradientes
            epsilon = error - nuevo_error
            error = nuevo_error
        return errores

        #gradient = self.gradient(images, Ts, gs)
        #print(gradient)

    # matchea imagen B en imagen A (osea mantenemos imagenA constante y transformamos B)
    def matchImages(self, imageA, imageB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(imageB.features, imageA.features, 2)
        matches = []
        
        for m in rawMatches:
            #  Lowe's ratio test
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        print("Cantidad de macheos: ", len(matches))
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([imageA.keypoints[i] for (i, _) in matches])
            ptsB = np.float32([imageB.keypoints[i] for (_, i) in matches])
            # compute the homography between the two sets of points
            inliners = np.zeros(max(len(ptsA), len(ptsB)))
            (H, inliners) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh, mask=inliners)
            
            return (matches, H, inliners)
        # otherwise, no homograpy could be computed
        else:
            raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        
    def error(self, images, Ts, gs):
        res = 0
        for i in np.arange(1, len(Ts)):
            for (f, gf) in gs[i-1]:
                Tf = np.dot(Ts[i-1], images[i-1].getKeypoint(f))
                Tpgf = np.dot(Ts[i], images[i].getKeypoint(gf))
                error = self.sqNorm(Tf - Tpgf)                
                res += error

        return res
    

    def gradient(self, images, Ts, gs):
        gradients = np.zeros((len(Ts), 3, 3))
        for i in range(0, len(images) - 1):
            for (fi, gfi) in gs[i-1]:
                try:
                    f = images[i-1].getKeypoint(fi)
                    gf = images[i].getKeypoint(gfi)
                    termino = np.dot(Ts[i-1], f) - np.dot(Ts[i], gf)
                    gradients[i] -= np.outer( (2 * termino), gf )
                    gradients[i-1] += np.outer( (2 * termino), f )
                except IndexError:
                    pass

        return gradients


    @staticmethod
    def sqNorm( v ):
        return np.dot(v, v)


if __name__ == '__main__':
    imagePath = 'Imagenes/'
    images = [cv2.imread(imagePath + 'fus1.tif'),
              cv2.imread(imagePath + 'fus2.tif')]

    stitcher = StitcherGD()
    errores = stitcher.stitch(images)
    
    plt.figure(0)
    plt.plot(np.arange(1,len(errores)+1), errores , linestyle='None', marker='.')
    plt.show()