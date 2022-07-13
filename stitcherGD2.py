# la intencion era buena... pero no anda
# posible solucion: calcular matrix homografica a manopla usando rotaciones-shear-escalado
# con CML por ejemplo

import numpy as np
import matplotlib.pyplot as plt
import cv2
from stitcher import Imagen
from ransac import findHomography


class StitcherGD:
    def stitch( self, images ,ratio=0.75, reprojThresh=4, retError=False ):
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

        Ts = self.normalize(Ts)

        errors = self.gradientDescent(Ts, images, gs, delta=0.000001, epsilon=0.5)

        if retError:
            return errors

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
            (H, inliners) = findHomography(ptsB, ptsA, reprojThresh)
            
            return (matches, H, inliners)
        # otherwise, no homograpy could be computed
        else:
            raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        
    def error(self, Ts, images, gs):
        res = 0
        for i in np.arange(1, len(Ts)):
            for (f, gf) in gs[i-1]:
                Tf = np.dot(Ts[i-1], images[i-1].getKeypoint(f))
                Tpgf = np.dot(Ts[i], images[i].getKeypoint(gf))
                error = self.sqNorm(Tf - Tpgf)
                                
                res += error

        return res

    def gradientDescent(self, Ts, images, gs, delta=0.001, epsilon=0.5):

        print(Ts)
        errors = [ np.finfo(np.float32).max,  self.error( Ts, images, gs)]
        while errors[-2] - errors[-1] > epsilon:
            gradient = self.getGradient(Ts, images, gs)
            print(gradient)
            for i in np.arange(1, len(Ts)):
                (p, r, q, sig, s, t) = self.getParameters(Ts[i])
                Ts[i] = self.createTransform( 
                    sig - delta * gradient['SIG'][i],
                    p - delta * gradient['P'][i],
                    r - delta * gradient['R'][i],
                    q - delta * gradient['Q'][i],
                    s - delta * gradient['S'][i],
                    t - delta * gradient['T'][i]
                )
            errors.append( self.error(Ts, images, gs) )
            print(Ts)

        return errors[1:]

    
    def getGradient(self, Ts, images, gs):
        
        # matrices gradientes
        gradient = {
            'P': [0],
            'R': [0],
            'Q': [0],
            'SIG': [0],
            'S': [0],
            'T': [0]
        }

        for i in range(1, len(images) ):
            self.append(gradient, 0)
            (p, r, q, sig, s, t) = self.getParameters(Ts[i])
            for (fi, gfi) in gs[i-1]:
                (fx, fy, _), (gfx, gfy, _) = images[i-1].getKeypoint(fi), images[i].getKeypoint(gfi)

                (phi1, phi2) = self.unproject(np.dot(Ts[i-1], images[i-1].getKeypoint(fi)))
                (lambda1, lambda2) = self.unproject(np.dot(Ts[i], images[i].getKeypoint(gfi)))

                expr1 = 2*(lambda1 - phi1)
                expr2 = 2*(lambda2 - phi2)

                self.updateParcialDerivatives(gradient, p, r, q, sig, fx, fy, gfx, gfy, expr1, expr2)

        for parameter in gradient:
            gradient[parameter][0] = 0

        return gradient

    @staticmethod
    def updateParcialDerivatives(gradient, p, r, q, sig, fx, fy, gfx, gfy, expr1, expr2):
        gradient['P'][-1] += expr1 * (-(np.cos(sig) * gfx + np.sin(sig) * gfy))
        gradient['R'][-1] += expr2 * (-q * (np.cos(sig) * gfx + np.sin(sig) * gfy) - np.cos(sig) * gfy + np.sin(sig) * gfx)
        gradient['Q'][-1] += expr2 * (-r * (np.cos(sig) * gfx + np.sin(sig) * gfy))
        gradient['SIG'][-1] += expr1 * (-p * (np.cos(sig) * gfy - np.sin(sig) * gfx)) + expr2 * (-r) * (q * (np.cos(sig) * gfy - np.sin(sig) * gfx) - np.sin(sig) * gfy - np.cos(sig) * gfx)
        gradient['S'][-1] += -expr1
        gradient['T'][-1] += -expr2

        try:
            gradient['P'][-2] += -expr1 * (np.cos(sig) * fx + np.sin(sig) * fy)
            gradient['R'][-2] += -expr2 * (q * (np.cos(sig) * fx + np.sin(sig) * fy) + np.cos(sig) * fy - np.sin(sig) * fx)
            gradient['Q'][-2] += -expr2 * (r * (np.cos(sig) * fx + np.sin(sig) * fy))
            gradient['SIG'][-2] += (-expr1) * (p * (np.cos(sig) * fy - np.sin(sig) * fx)) + (-expr2) * r * (q * (np.cos(sig) * fy - np.sin(sig) * fx) - np.sin(sig) * fy - np.cos(sig) * fx)
            gradient['S'][-2] += -expr1
            gradient['T'][-2] += -expr2 
        except IndexError:
            pass


    @staticmethod
    def sqNorm( v ):
        return np.dot(v, v)

    @staticmethod
    def unproject( coord ):
        return coord[0:2] / coord[2]
    
    @staticmethod
    def getParameters( T ):
        (Q,R) = np.linalg.qr(T[0:2, 0:2])

        sigma = np.arctan2(T[0][1], T[0][0])
        p = R[0][0]
        q = R[0][1] / p
        r = R[1][1]

        return (p, r, q, sigma, T[0][2], T[1][2] )

        

    @staticmethod
    def normalize( Ts ):
        #for T in Ts:            
        #    T[1] -= T[0] * (T[1][0] / T[0][0])
        #    T[2] -= T[0] * (T[2][0] / T[0][0])
        #    T[2] -= T[1] * (T[2][1] / T[1][1])
        #    T /= T[2][2]
        return np.array(Ts)

    
    @staticmethod
    def append( vs , value ):
        for v in vs:
            vs[v].append(value)

    @staticmethod
    def createTransform( sigma, sx, sy, sh, tx, ty ):
        scale = np.array([ [ sx, 0 ], [0, sy ] ] )
        shear = np.array([ [ 1, sh ] , [ 0, 1] ] )
        rotation = np.array([ [-np.cos(sigma), np.sin(sigma) ], [np.sin(sigma), np.cos(sigma)]])

        return np.block( [ [ np.dot( rotation, np.dot(shear, scale) ), np.array([[tx],[ty]]) ], [np.zeros((1,2)), 1 ] ] )

if __name__ == '__main__':
    imagePath = 'Imagenes/'
    images = [cv2.imread(imagePath + 'fus1.tif'),
              cv2.imread(imagePath + 'fus2.tif')]

    stitcher = StitcherGD()
    errors = stitcher.stitch(images, retError=True)

    print("ERRORES: ", errors)
    #plt.figure(0)
    #plt.plot(np.arange(1,len(errors) + 1), errors, color='blue', linestyle='None', marker='.')
    #plt.show()
    