import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import cv2
from numpy.core.fromnumeric import shape
from imagen import Imagen
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def fittear( ptData, ptTrain ):
    return np.float32( [
        [1, 0, ptTrain[0] - ptData[0]],
        [0, 1, ptTrain[1] - ptData[1]],
        [0, 0, 1]
    ] )

def calcularInliers( ptsData, ptsTrain, H, tolerancia):
    ptsData = np.block([ ptsData, np.ones( ( ptsData.shape[0], 1 )) ])
    ptsTrain = np.block([ ptsTrain, np.ones( ( ptsTrain.shape[0], 1 )) ])
    E = H@ptsData.T - ptsTrain.T

    return np.sum( [ np.dot( e,e ) < np.power(tolerancia, 2) for e in E.T  ] )

def matrizHomografica( ptsData, ptsTrain, tolerancia, maxIteraciones ):
    mejorFitteo = (np.eye(3), -np.inf) # matriz asociada y inliners
    for i in range(maxIteraciones):
        indiceInicial = np.random.randint(0, len(ptsData))
        H = fittear(ptsData[indiceInicial], ptsTrain[indiceInicial])
        cantInliers = calcularInliers(ptsData, ptsTrain, H, tolerancia)

        mejorFitteo = (H, cantInliers) if cantInliers > mejorFitteo[1] else mejorFitteo

    return mejorFitteo

def matchearImagenes(imagenA, imagenB, ratio, tolerancia, maxIteracionesRansac, mascaras=None):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
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
        (H, inliners) = matrizHomografica( ptsB, ptsA, tolerancia, maxIteracionesRansac)

        if H is None:
            raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        return (matches, H, inliners)
    else:
        raise Exception("No hubo suficientes coincidencias entre las caracteristicas de las imagenes")
        
def stitchPar(imagenA, imagenB, ratio, tolerancia, maxIteracionesRansac, mascaras=None, limite=None):
    (matches, H, inliers) = matchearImagenes(imagenA, imagenB, ratio, tolerancia, maxIteracionesRansac, mascaras)
    
    imagenB.mover(H[0][2], H[1][2])
    
    smallestX = np.min([imagenB.posicion[0], imagenA.posicion[0]])
    biggestX = np.max([imagenB.posicion[0] + imagenB.shape[1],
                      imagenA.posicion[0] + imagenA.shape[1]])
    smallestY = np.min([imagenB.posicion[1], imagenA.posicion[1]])
    biggestY = np.max([imagenB.posicion[1] + imagenB.shape[0],
                      imagenA.posicion[1] + imagenA.shape[0]])

    resultShape = (
        int(np.floor(biggestY - smallestY)),
        int(np.floor(biggestX - smallestX)),
        3
    )

    result = Imagen(np.zeros(resultShape, np.uint8), imagenA.descriptor, np.zeros(resultShape[0:2], np.uint8) )
    result.pegar(imagenA)
    result.pegar(imagenB)

    return result

def stitch( imagenes, grilla , maxIteracionesRansac, ratio, tolerancia, path, ratioMascaraW=0.3, ratioMascaraH=0.5 ):
    imagenes = deque(imagenes)
    ultimaFilaX, ultimaFilaAlto, panoramaShape = 0, 0, (0,0)
    for fila in range(grilla[0]):
        imagenFila = imagenes.popleft()
        anchoUltimaPegada = imagenFila.shape[1]
        for col in range(grilla[1] - 1):
            imagenAPegar = imagenes.popleft()
            mascaras = [ 
                Imagen.crearMascara(imagenFila.shape, (anchoUltimaPegada * ratioMascaraW) / imagenFila.shape[1] if col != 0 else ratioMascaraW),
                Imagen.crearMascara(imagenAPegar.shape, ratioMascaraW, orden=1)
            ]
            anchoUltimaPegada = imagenAPegar.shape[1]
            imagenFila = stitchPar( imagenFila, imagenAPegar, ratio, tolerancia, maxIteracionesRansac, mascaras=mascaras )
        
        if fila == 0:
            panoramaShape = imagenFila.shape
            panorama = np.memmap(path, dtype='uint8', mode='w+', shape=panoramaShape)
            panorama[:] = imagenFila.imagen[:]
            ultimaFilaAlto = panoramaShape[0]
        else:
            roiPanorama = Imagen(np.memmap(path, dtype='uint8', mode='r+', shape=( int(ultimaFilaAlto * ratioMascaraH), panoramaShape[1], 3), offset=int(ultimaFilaX * panoramaShape[1] * 8)), imagenFila.descriptor)
            mascaras = [ roiPanorama.mascara, Imagen.crearMascara(imagenFila.shape, ratioMascaraH, eje=1, orden=1)]
            (matches, H, inliers) = matchearImagenes(roiPanorama, imagenFila, ratio, tolerancia, maxIteracionesRansac, mascaras)
            
            shapeFinal = ( int(H[1][2]) + imagenFila.shape[0], panoramaShape[1], 3 )
            panoramaShape = ( int(panoramaShape[0] + shapeFinal[0] - ultimaFilaAlto * ratioMascaraH), panoramaShape[1], panoramaShape[2])
            ultimaFilaX += H[1][2]
            ultimaFilaAlto = imagenFila.shape[0]

            imagenFila.transformar( H, shape=(shapeFinal[0], shapeFinal[1]) )
            result = Imagen(np.zeros(shapeFinal, np.uint8), imagenFila.descriptor, np.zeros(shapeFinal[0:2], np.uint8) )
            result.pegar(roiPanorama)
            result.pegar(imagenFila)

            panorama = np.memmap(path, dtype='uint8', mode='r+', shape=shapeFinal, offset=int( ultimaFilaX * panoramaShape[1] * 8) )
            panorama[:] = result.imagen[:]

    return np.memmap(path, dtype='uint8', mode='r', shape=panoramaShape)
        

if __name__ == '__main__':
    #tracemalloc.start()

    descriptor = cv2.xfeatures2d.SURF_create()
    grilla = (2,4)
    orden = [ i if int((i-1)/4) % 2 == 0 else 4*(int((i-1)/4)+1) - (i-1)%4 for i in range(1, grilla[0] * grilla[1] + 1)]

    ratio = 0.75
    tolerancia = 4

    # leemos las imagenes
    res = stitch(( Imagen( cv2.imread('Imagenes/Intestino curso/' + f"{i:02d}" + '.tif'), descriptor ) for i in orden ),
            grilla,
            maxIteracionesRansac=8,
            ratio=ratio,
            tolerancia=tolerancia,
            path="panorama_2x4.nparr")

    plt.imshow(res[...,::-1])
    plt.show()

    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)