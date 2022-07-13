import numpy as np
import cv2


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

        self._descriptor = descriptor
        self._describida = False

    @property
    def keypoints(self):
        if self._describida:
            return self._keypoints
        else:
            self.detectarYDescribir()
            return self._keypoints

    @property
    def features(self):
        if self._describida:
            return self._features
        else:
            self.detectarYDescribir()
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

    def detectarYDescribir(self):
        gray = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)

        (kps, features) = self._descriptor.detectAndCompute(gray, mask=None)

        self._keypoints = np.float32([kp.pt for kp in kps])
        self._features = features
        self._describida = True

    def transformar(self, matrizHomografica):

        self._esquinas = (np.dot(matrizHomografica, self._esquinas.T)).T

        smallestX = np.min([esquina[0] for esquina in self._esquinas])
        biggestX = np.max([esquina[0] for esquina in self._esquinas])
        smallestY = np.min([esquina[1] for esquina in self._esquinas])
        biggestY = np.max([esquina[1] for esquina in self._esquinas])

        shape = (
            int(np.floor(biggestY - smallestY)),
            int(np.floor(biggestX - smallestX))
        )

        # corrigo para que quede en 0,0
        nuevaHomografica = np.add(matrizHomografica, [
            [0, 0, -smallestX],
            [0, 0, -smallestY],
            [0, 0, 0]
        ])

        self.imagen = cv2.warpPerspective(self.imagen, nuevaHomografica, (
            shape[1], shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(125, 125, 125, 0))

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


class Stitcher:
    def __init__(self):
        self._descriptor = cv2.xfeatures2d.SURF_create()

    def stitch(self, images, size, panningOrder=None, ratio=0.75, reprojThresh=4):
        if len(images) < 2:
            raise Exception("Se necesitan al menos dos imagenes")

        if panningOrder is None:
            panningOrder = np.arange(1, len(images) + 1, 1, dtype=np.int16)

        if len(panningOrder) != size[0] * size[1]:
            raise Exception("El orden no coincide con el tamaño del panorama")

        # las paso a mi clase Image
        images = [Imagen(image, self._descriptor) for image in images]

        # momento de decisiones... primero pego por columnas y despues uno las filas??
        print("Comenzando pegado...")
        index = 0
        panorama = None
        for row in range(size[0]):
            currentRowImage = images[panningOrder[index] - 1]
            for col in range(size[1] - 1):
                currentRowImage = self.stitchPair(
                    currentRowImage, images[panningOrder[index + 1] - 1], ratio, reprojThresh)
                index += 1
            index += 1
            panorama = currentRowImage if panorama is None else self.stitchPair(
                panorama, currentRowImage, ratio, reprojThresh)

        return panorama.imagen

    def stitchPair(self, imageA, imageB, ratio, reprojThresh):
        M = self.matchImages(imageA, imageB, ratio, reprojThresh)

        (matches, H, status) = M
        imageB.transformar(H)

        imageA.mover(int(np.abs(np.clip(np.min([corner[0] for corner in imageB.esquinas]), None, 0))),
                    int(np.abs(np.clip(np.min([corner[1] for corner in imageB.esquinas]), None, 0))))

        smallestX = np.min([imageB.posicion[0], imageA.posicion[0]])
        biggestX = np.max([imageB.posicion[0] + imageB.shape[1],
                          imageA.posicion[0] + imageA.shape[1]])
        smallestY = np.min([imageB.posicion[1], imageA.posicion[1]])
        biggestY = np.max([imageB.posicion[1] + imageB.shape[0],
                          imageA.posicion[1] + imageA.shape[0]])

        resultShape = (
            int(np.floor(biggestY - smallestY)),
            int(np.floor(biggestX - smallestX)),
            4
        )
        result = Imagen(np.zeros(resultShape, np.uint8), self._descriptor)

        # pego imagen central
        result.pegar(imageA)
        result.pegar(imageB)

        return result

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
            (H, status) = cv2.findHomography(
                ptsB, ptsA, cv2.RANSAC, reprojThresh)

            return (matches, H, status)
        # otherwise, no homograpy could be computed
        else:
            raise Exception(
                "No hubo suficientes coincidencias entre las caracteristicas de las imagenes")


if __name__ == '__main__':
    imagePath = 'Imagenes/Intestino curso'
    images = []
    for i in range(2, 4):
        images.append(cv2.imread(imagePath + '/0' + str(i) + '.tif'))

    if any([image is None for image in images]):
        raise Exception("Alguna de las direcciones de imagenes no existe")

    stitcher = Stitcher()
    res = stitcher.stitch(images, size=(1, 2))
    cv2.imwrite("ejemplos/resultIntestino.jpg", res)
