import numpy as np
import cv2

class Imagen:
    def __init__(self, image, descriptor, mascara=None):
        self.imagen = image

        self._keypoints = None
        self._features = None
        self._posicion = (0, 0) # esquina sup izq en fondo

        self.descriptor = descriptor
        self._descripta = False
        
        self.mascara = np.ones( self.imagen.shape[0:2], dtype=np.uint8 ) * 255 if mascara is None else mascara

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

    def transformar( self, matrizHomografica, shape):
        roi = self.calcularROI( matrizHomografica, shape )

        self.imagen = cv2.warpPerspective(self.imagen, matrizHomografica, (shape[1], shape[0]))     
        self.mascara = cv2.warpPerspective(self.mascara, matrizHomografica, (shape[1], shape[0]))

        return roi

    def calcularROI( self, H, newShape ):
        esquinas = np.array( [
            [0, 0, 1],
            [self.shape[1], 0, 1],
            [0, self.shape[0], 1],
            [self.shape[1], self.shape[0], 1] ] )
        
        roi = (H @ esquinas.T).T
        roi = np.array([ esquina[0:2] / esquina[2] for esquina in roi] )
        roi = np.clip( roi, np.full((4,2),[0,0]), np.full((4,2), [newShape[1], newShape[0]]))
        return roi

    def mover(self, x, y):
        self._posicion = (x, y)

    def pegar(self, imagenAPegar):
        if imagenAPegar.posicion[0] > self.imagen.shape[1] or imagenAPegar.posicion[1] > self.imagen.shape[0]:
            return

        Traslacion = np.float32( [ 
            [1, 0, imagenAPegar.posicion[0]], 
            [0, 1, imagenAPegar.posicion[1]], 
            [0,0,1] 
        ])
        imagenTrasladada = cv2.warpPerspective(imagenAPegar.imagen, Traslacion, (self.imagen.shape[1], self.imagen.shape[0]))
        mascaraTrasladada = cv2.warpPerspective(imagenAPegar.mascara, Traslacion, (self.imagen.shape[1], self.imagen.shape[0]))
        invMascaraTrasladada = cv2.bitwise_not(mascaraTrasladada)
        
        # le borro donde pego la otra imagen
        self.imagen = cv2.bitwise_and( self.imagen, self.imagen, mask=invMascaraTrasladada)
        cv2.add( self.imagen, imagenTrasladada, dst=self.imagen)
        
        cv2.bitwise_or( self.mascara, mascaraTrasladada, dst=self.mascara )
            
    # eje marca si de izq a der o arriba a abajo
    # orden si a izq o a der (o arr o abajo)
    @staticmethod
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
