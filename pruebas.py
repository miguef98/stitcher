import numpy as np
import cv2
import matplotlib.pyplot as plt

H = np.array([[ 9.84754111e-01, 5.72847757e-04,  5.02771853e+02],
              [-5.66063673e-03, 9.96065585e-01, -1.90808549e+00],
              [-1.06520777e-05, 1.03415334e-06,  1.00000000e+00]])

imagen = cv2.imread("Imagenes/Intestino curso/01.tif")

esquinas = np.array([
        [0, 0, 1],
        [imagen.shape[1], 0, 1],
        [0, imagen.shape[0], 1],
        [imagen.shape[1], imagen.shape[0], 1]
    ])
esquinas = (np.dot(H, esquinas.T)).T
esquinas = np.array([ esquina / esquina[2] for esquina in esquinas ])

smallestX = np.min([esquina[0] for esquina in esquinas])
biggestX = np.max([esquina[0] for esquina in esquinas])
smallestY = np.min([esquina[1] for esquina in esquinas])
biggestY = np.max([esquina[1] for esquina in esquinas])

nuevaShape = (
    int(np.floor(biggestY - smallestY)),
    int(np.floor(biggestX - smallestX))
)

traslacion = np.array([
    [1, 0, -smallestX],
    [0, 1, -smallestY],
    [0, 0, 1]
])
nuevaHomografica = np.matmul(traslacion, H)

imagen2 = cv2.warpPerspective(imagen, nuevaHomografica, (nuevaShape[1], nuevaShape[0]))

plt.imshow(imagen2)
plt.show()

