import numpy as np
import cv2

class Imagen:
    def __init__(self, image, descriptor, mascara=None, esVacia=False):
        self.imagen = image

        self._keypoints = None
        self._features = None
        self._posicion = (0, 0) # esquina sup izq en fondo
        self._esquinas = np.array([
            [0, 0, 1],
            [self.imagen.shape[1], 0, 1],
            [0, self.imagen.shape[0], 1],
            [self.imagen.shape[1], self.imagen.shape[0], 1]
        ]) if not esVacia else None

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

    def transformar( self, matrizHomografica, shape=None):
        self._esquinas = (np.dot(matrizHomografica, self._esquinas.T)).T
        self._esquinas = np.array([ esquina / esquina[2] for esquina in self._esquinas ])

        if shape is None:
            self.transformarCentrado(matrizHomografica)
        else:
            self.transformarLibre(matrizHomografica, shape)
        
        self._descripta = False

    def transformarCentrado(self, matrizHomografica):
        smallestX = np.min([esquina[0] for esquina in self._esquinas])
        biggestX = np.max([esquina[0] for esquina in self._esquinas])
        smallestY = np.min([esquina[1] for esquina in self._esquinas])
        biggestY = np.max([esquina[1] for esquina in self._esquinas])

        nuevaShape = (
            int(np.floor(biggestY - smallestY)),
            int(np.floor(biggestX - smallestX))
        )

        # corrijo para que quede en 0,0
        traslacion = np.array([
            [1, 0, -smallestX],
            [0, 1, -smallestY],
            [0, 0, 1]
        ])
        nuevaHomografica = np.matmul(traslacion, matrizHomografica)

        self.imagen = cv2.warpPerspective(self.imagen, nuevaHomografica, (nuevaShape[1], nuevaShape[0]))
        
        self.mascara = cv2.warpPerspective(self.mascara, nuevaHomografica, (nuevaShape[1], nuevaShape[0]))

        self._posicion = [0 if smallestX < 0 else int(
            np.round(smallestX)), 0 if smallestY < 0 else int(np.round(smallestY))]
    
    def transformarLibre( self, matrizHomografica, shape ):
        self.imagen = cv2.warpPerspective(self.imagen, matrizHomografica, (shape[1], shape[0]))     
        self.mascara = cv2.warpPerspective(self.mascara, matrizHomografica, (shape[1], shape[0]))

    def mover(self, x, y):
        self._posicion = (x, y)

    def pegar(self, imagenAPegar):
        if imagenAPegar.posicion[0] > self.imagen.shape[1] or imagenAPegar.posicion[1] > self.imagen.shape[0]:
            return

        self.actualizarEsquinas( imagenAPegar )

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
   
    def actualizarEsquinas( self, imagenAPegar ):
        if self._esquinas is None:
           self._esquinas = np.hstack( [imagenAPegar.esquinas + np.full( (4,1) , [imagenAPegar.posicion[0], imagenAPegar.posicion[1]] ), np.ones((3,1))])
        else:
            esquinasTotales = np.concatenate( [self.esquinas, imagenAPegar.esquinas + np.full( (4,1) , [imagenAPegar.posicion[0], imagenAPegar.posicion[1]] ) ])

            smallestX = np.min([esquina[0] for esquina in esquinasTotales])
            biggestX = np.max([esquina[0] for esquina in esquinasTotales])
            smallestY = np.min([esquina[1] for esquina in esquinasTotales])
            biggestY = np.max([esquina[1] for esquina in esquinasTotales])

            self._esquinas = [ 
                [smallestX, smallestY, 1],
                [biggestX, smallestY, 1],
                [smallestX, biggestY, 1],
                [biggestX, biggestY, 1]
            ]
    
    def histogramaCDFs( self, bins=256, rango=(0,256), mascara=None ):
        canales = cv2.split( self.imagen )
    
        if mascara is None:
            histogramas = [ np.histogram(canal, bins=bins, range=rango, density=True)[0] for canal in canales ]
        else:
            histogramas = [ np.histogram(canal[mascara == 255], bins=bins, range=rango, density=True)[0] for canal in canales ]
        
        acumuladas = [ np.add.accumulate(histograma) for histograma in histogramas ]
        return acumuladas
    
    def matchearHistogramas( self, imagenAMatchear ):
        TraslacionA = np.float32( [ 
            [1, 0, -self.posicion[0]], 
            [0, 1, -self.posicion[1]], 
            [0,0,1] 
        ])
        
        TraslacionB = np.float32( [ 
            [1, 0, self.posicion[0]], 
            [0, 1, self.posicion[1]], 
            [0,0,1] 
        ])

        mascaraBTrasladada = cv2.warpPerspective(self.mascara, TraslacionB, (imagenAMatchear.shape[1], imagenAMatchear.shape[0]))
        superposicionTarget = cv2.bitwise_and(imagenAMatchear.mascara, mascaraBTrasladada)

        mascaraATrasladada = cv2.warpPerspective( imagenAMatchear.mascara, TraslacionA, (self.shape[1], self.shape[0]) )
        superposicionReference = cv2.bitwise_and(self.mascara, mascaraATrasladada)
        
        histogramasPropios = self.histogramaCDFs( mascara=superposicionReference )
        histogramasTarget = imagenAMatchear.histogramaCDFs( mascara=superposicionTarget )
        
        # funcion de matcheo por canal
        MBlue = np.zeros(256, dtype=np.uint8)
        MGreen = np.zeros(256, dtype=np.uint8)
        MRed = np.zeros(256, dtype=np.uint8)

        for G1 in np.arange(0,256):
            G2 = np.searchsorted( histogramasTarget[0], histogramasPropios[0][G1], side="left" )
            MBlue[G1] = min( np.uint8( np.round(G2) ), 255)

            G2 = np.searchsorted( histogramasTarget[1], histogramasPropios[1][G1], side="left" )
            MGreen[G1] = min( np.uint8( np.round(G2) ), 255)

            G2 = np.searchsorted( histogramasTarget[2], histogramasPropios[2][G1], side="left" )
            MRed[G1] = min( np.uint8( np.round(G2) ), 255)
            
        canales_B = cv2.split( self.imagen )
        nuevoB = np.array( [ MBlue[b] for b in canales_B[0] ] )
        nuevoG = np.array( [ MGreen[g] for g in canales_B[1] ] )
        nuevoR = np.array( [ MRed[r] for r in canales_B[2] ] )

        self.imagen = cv2.merge( [nuevoB, nuevoG, nuevoR], self.imagen)
        
    def matchearHistogramasSinMascaras( self, imagenAMatchear ):        
        histogramasPropios = self.histogramaCDFs( mascara=self.mascara )
        histogramasTarget = imagenAMatchear.histogramaCDFs( mascara=imagenAMatchear.mascara )
        
        # funcion de matcheo por canal
        MBlue = np.zeros(256, dtype=np.uint8)
        MGreen = np.zeros(256, dtype=np.uint8)
        MRed = np.zeros(256, dtype=np.uint8)

        for G1 in np.arange(0,256):
            G2 = np.searchsorted( histogramasTarget[0], histogramasPropios[0][G1], side="left" )
            MBlue[G1] = min( np.uint8( np.round(G2) ), 255)

            G2 = np.searchsorted( histogramasTarget[1], histogramasPropios[1][G1], side="left" )
            MGreen[G1] = min( np.uint8( np.round(G2) ), 255)

            G2 = np.searchsorted( histogramasTarget[2], histogramasPropios[2][G1], side="left" )
            MRed[G1] = min( np.uint8( np.round(G2) ), 255)
            
        canales_B = cv2.split( self.imagen )
        nuevoB = np.array( [ MBlue[b] for b in canales_B[0] ] )
        nuevoG = np.array( [ MGreen[g] for g in canales_B[1] ] )
        nuevoR = np.array( [ MRed[r] for r in canales_B[2] ] )

        self.imagen = cv2.merge( [nuevoB, nuevoG, nuevoR], self.imagen)
    
    def memmap( self, filepath ):
        fp = np.memmap( filepath, dtype='uint8', mode='w+', shape=self.imagen.shape )
        fp[:] = self.imagen[:]
        self.imagen = fp 
        
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
