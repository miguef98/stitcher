import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 7]

class Imagen:
    def __init__(self, image, descriptor, canalAlpha=True, alpha=255):
        self.imagen = image
        if canalAlpha:
            self.agregarCanalAlpha(alpha)

        self._keypoints = None
        self._features = None
        self._posicion = (0, 0)
        self._esquinas = np.array([
            [0, 0, 1],
            [self.imagen.shape[1], 0, 1],
            [0, self.imagen.shape[0], 1],
            [self.imagen.shape[1], self.imagen.shape[0], 1]
        ])

        self.descriptor = descriptor
        self._descripta = False

    @property
    def keypoints(self):
        if self._descripta:
            return np.float32([kp.pt for kp in self._keypoints])
        else:
            self.detectarYDescribir(None)
            return np.float32([kp.pt for kp in self._keypoints])

    @property
    def features(self):
        if self._descripta:
            return self._features
        else:
            self.detectarYDescribir(None)
            return self._features
    
    def getKeypoints(self, mascara=None):
        self.detectarYDescribir( mascara )
        return np.float32([kp.pt for kp in self._keypoints])

    def getFeatures(self, mascara=None):
        self.detectarYDescribir( mascara )
        return self._features

    @property
    def esquinas(self):
        return np.round(self._esquinas[:, 0:2])

    @property
    def posicion(self):
        return self._posicion

    @property
    def shape(self):
        return self.imagen.shape

    def detectarYDescribir(self, mascara):
        gray = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        
        (kps, features) = self.descriptor.detectAndCompute(gray, mask=mascara)

        self._keypoints = kps
        self._features = features
        self._descripta = True

    def transformar(self, matrizHomografica, limite=None):
        esquinasViejas = self._esquinas
        self._esquinas = (np.dot(matrizHomografica, self._esquinas.T)).T
        self._esquinas = np.array([ esquina / esquina[2] for esquina in self._esquinas ])

        smallestX = np.min([esquina[0] for esquina in self._esquinas])
        biggestX = np.max([esquina[0] for esquina in self._esquinas])
        smallestY = np.min([esquina[1] for esquina in self._esquinas])
        biggestY = np.max([esquina[1] for esquina in self._esquinas])

        nuevaShape = (
            int(np.floor(biggestY - smallestY)),
            int(np.floor(biggestX - smallestX))
        )

        if not limite is None and (2 * self.shape[0] < nuevaShape[0] or 2 * self.shape[1] < nuevaShape[1]):
            self._esquinas = esquinasViejas
            raise Exception("La transformacion excede limite de tamano")

        # corrijo para que quede en 0,0
        traslacion = np.array([
            [1, 0, -smallestX],
            [0, 1, -smallestY],
            [0, 0, 1]
        ])
        nuevaHomografica = np.matmul(traslacion, matrizHomografica)

        self.imagen = cv2.warpPerspective(self.imagen, nuevaHomografica, (
            nuevaShape[1], nuevaShape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(125, 125, 125, 0))

        self._posicion = [0 if smallestX < 0 else int(
            np.round(smallestX)), 0 if smallestY < 0 else int(np.round(smallestY))]

        self._describida = False

    def mover(self, x, y):
        self._posicion = (x, y)

    def agregarCanalAlpha(self, value):
        if self.imagen.shape[2] < 4:
            b_channel, g_channel, r_channel = cv2.split(self.imagen)
            alpha_channel = np.ones(
                b_channel.shape, dtype=b_channel.dtype) * value
            self.imagen = cv2.merge(
                (b_channel, g_channel, r_channel, alpha_channel))

    def pegar(self, imagenAPegar):
        if imagenAPegar.posicion[0] > self.imagen.shape[1] or imagenAPegar.posicion[1] > self.imagen.shape[0]:
            return

        else:
            h, w = imagenAPegar.shape[0], imagenAPegar.shape[1]
            (x, y) = imagenAPegar.posicion
            if x+w > self.imagen.shape[1] or y+h > self.imagen.shape[0]:
                raise Exception("La imagen a pegar supera el tamaño del fondo")

            overlay_image = imagenAPegar.imagen
            mask = imagenAPegar.imagen[..., 3:] / 255.0
            self.imagen[y:y+h, x:x+w] = (1.0 - mask) * \
                self.imagen[y:y+h, x:x+w] + mask * overlay_image

# eje marca si de izq a der o arriba a abajo
# orden si a izq o a der (o arr o abajo)
def crearMascara( shape, proporcion, eje=0, orden=0 ):
    if eje == 0:
        anchoMascara = int( shape[1] * proporcion )
        izq = np.zeros( (shape[0],shape[1] - anchoMascara, 1), dtype=np.uint8 )
        der = np.ones( (shape[0], anchoMascara, 1), dtype=np.int8 ) * 255
        return np.hstack( (izq,der) ).astype(np.uint8) if orden == 0 else np.hstack( (der,izq) ).astype(np.uint8)
    else:
        altoMascara = int( shape[0] * proporcion )
        arr = np.zeros( (shape[0] - altoMascara, shape[1], 1), dtype=np.uint8 )
        aba = np.ones( (altoMascara, shape[1], 1), dtype=np.int8 ) * 255
        return np.vstack( (arr,aba) ).astype(np.uint8) if orden == 0 else np.vstack( (aba,arr) ).astype(np.uint8)
    
def dibujarMatcheos( imagenA, imagenB, mascaras ):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(imagenA.getFeatures( mascaras[0] ),
                                  imagenB.getFeatures( mascaras[1] ),
                                  2)

    matchesMask = [[0,0] for i in range(len(rawMatches))]
    imagen1Matcheos = imagenA.imagen.copy()
    imagen2Matcheos = imagenB.imagen.copy()
    for i, (m1,m2) in enumerate(rawMatches):
        if m1.distance < ratio * m2.distance:
            matchesMask[i] = [1,0]
            pt1 = imagenA.keypoints[m1.queryIdx]
            pt2 = imagenB.keypoints[m1.trainIdx]
            cv2.circle(imagen1Matcheos, (int(pt1[0]),int(pt1[1])), 10, (255,0,255), -1)
            cv2.circle(imagen2Matcheos, (int(pt2[0]),int(pt2[1])), 10, (255,0,255), -1)


    ## Draw match in blue, error in red
    draw_params = dict(matchColor = (255, 0,0),
                       singlePointColor = (0,0,255),
                       matchesMask = matchesMask,
                       flags = 0)

    res = cv2.drawMatchesKnn(imagen1Matcheos,
                             imagenA._keypoints,
                             imagen2Matcheos,
                             imagenB._keypoints,
                             rawMatches,
                             None,
                             **draw_params)
    return res


def matchearImagenes(imagenA, imagenB, ratio, tolerancia, mascaras=None):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    if mascaras is None:
        rawMatches = matcher.knnMatch(imagenB.features, imagenA.features, 2)
    else:
        rawMatches = matcher.knnMatch(imagenB.getFeatures(mascaras[1]), imagenA.getFeatures(mascaras[0]), 2)
    matches = []

    for m in rawMatches:
        #  Lowe's ratio test
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        ptsA = np.float32([imagenA.keypoints[i] for (i, _) in matches])
        ptsB = np.float32([imagenB.keypoints[i] for (_, i) in matches])
        (H, inliners) = cv2.findHomography( ptsB, ptsA, cv2.RANSAC, tolerancia)

        if H is None:
            raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        return (matches, H, inliners)
    else:
        raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        
def stitchPar(imagenA, imagenB, ratio, tolerancia, mascaras=None, deb=False):
    (matches, H, inliners) = matchearImagenes(imagenA, imagenB, ratio, tolerancia, mascaras)
    
    try:
        imagenB.transformar(H, limite=2)
    except Exception as e:
        plt.imshow(imagenB.imagen)
        plt.show()
        raise(e)

    if deb:
        plt.imshow(imagenB.imagen)
        plt.show()

    imagenA.mover(int(np.abs(np.clip(np.min([esquina[0] for esquina in imagenB.esquinas]), None, 0))),
                  int(np.abs(np.clip(np.min([esquina[1] for esquina in imagenB.esquinas]), None, 0))))

    smallestX = np.min([imagenB.posicion[0], imagenA.posicion[0]])
    biggestX = np.max([imagenB.posicion[0] + imagenB.shape[1],
                      imagenA.posicion[0] + imagenA.shape[1]])
    smallestY = np.min([imagenB.posicion[1], imagenA.posicion[1]])
    biggestY = np.max([imagenB.posicion[1] + imagenB.shape[0],
                      imagenA.posicion[1] + imagenA.shape[0]])

    resultShape = (
        int(np.floor(biggestY - smallestY)),
        int(np.floor(biggestX - smallestX)),
        4
    )
    result = Imagen(np.zeros(resultShape, np.uint8), imagenA.descriptor )

    result.pegar(imagenA)
    result.pegar(imagenB)

    return result

def stitch( imagenes, grilla , orden ):
    indice = 0
    panorama = None
    for fila in range(grilla[0]):
        imagenFila = imagenes[orden[indice] - 1]
        for col in range(grilla[1] - 1):
            mascaras = (
                        crearMascara(imagenFila.shape, (1/ (col+1)) / 2 ),
                        crearMascara(imagenes[orden[indice + 1] - 1].shape, 0.5, orden=1)
            )
            if col == 1:
                imagenFila = stitchPar( imagenFila, imagenes[orden[indice + 1] - 1], ratio, tolerancia, mascaras, deb=True )
            else:
                imagenFila = stitchPar( imagenFila, imagenes[orden[indice + 1] - 1], ratio, tolerancia, mascaras)

            indice += 1
        indice += 1
        panorama = imagenFila if panorama is None else stitchPar( panorama, imagenFila, ratio, tolerancia)

    return panorama.imagen

descriptor = cv2.SIFT_create()
grilla = (1,4)
orden = [1,2,3,4]

ratio = 0.75
tolerancia = 4

# leemos las imagenes
imagenes = [ Imagen( cv2.imread('Imagenes/Intestino curso/01.tif'), descriptor ),
             Imagen( cv2.imread('Imagenes/Intestino curso/02.tif'), descriptor ),
             Imagen( cv2.imread('Imagenes/Intestino curso/03.tif'), descriptor ),
             Imagen( cv2.imread('Imagenes/Intestino curso/04.tif'), descriptor )]

resultado = stitch(imagenes, grilla, orden)

plt.imshow(resultado)
plt.show()