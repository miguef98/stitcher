import numpy as np
import cv2

def absOrZero( number ):
	if number < 0:
		return np.abs(number)
	else:
		return 0

class Stitcher:
	def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageA, imageB) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		# match features between the two images
		M = self.matchKeypoints(kpsB, kpsA,
								featuresB, featuresA, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		(imageBT, corners, positionImageB) = self.transform(imageB, H)

		positionImageA = [int(absOrZero( np.min( [corner[0] for corner in corners] ) )),
						  int(absOrZero( np.min( [corner[1] for corner in corners] ) ))]


		smallestX = np.min( [positionImageB[0], positionImageA[0]] )
		biggestX = np.max( [positionImageB[0] + imageBT.shape[1] ,positionImageA[0] + imageA.shape[1]] )
		smallestY = np.min( [positionImageB[1], positionImageA[1]])
		biggestY = np.max( [positionImageB[1] + imageBT.shape[0] ,positionImageA[1] + imageA.shape[0]])

		resultShape = (
			int(biggestY - smallestY), 
			int(biggestX - smallestX),
			3
		)
		result = np.zeros(resultShape, np.uint8)

		#pego imagen central
		self.paste(result, imageA, positionImageA[0], positionImageA[1])

		# pego imagen transformada
		self.paste(result, imageBT, positionImageB[0], positionImageB[1])


		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageB, imageA, kpsB, kpsA, matches,
									status)
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)
		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect and extract features from the image
		descriptor = cv2.xfeatures2d.SURF_create()  # cv2.SURF_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
												reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 4), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
		return vis

	def transform(self, image, transform):
		# la escribo normal y despues la transpongo para que sea mas facil de leer
		# las filas aca son las esquinas
		originalCorners = np.array([
			[0, 0, 1],
			[image.shape[1], 0, 1],
			[0, image.shape[0], 1],
			[image.shape[1], image.shape[0], 1]
		])

		resCorners = (np.round(np.dot(transform, originalCorners.T))[0:-1, :]).T
		shape = (
			int(np.ceil(max(resCorners[2][1], resCorners[3][1]) - min(resCorners[0][1], resCorners[0][1]))),
			int(np.ceil(max(resCorners[1][0], resCorners[3][0]) - min(resCorners[0][0], resCorners[2][0])))
		)

		smallestX = np.min([corner[0] for corner in resCorners])
		smallestY = np.min([corner[1] for corner in resCorners])
		# corrigo para que quede en 0,0
		transform[0][2] += -1 * smallestX
		transform[1][2] += -1 * smallestY
		
		result = cv2.warpPerspective( image, transform , (shape[1], shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))

		# posicion en resultado de la imagen transformada:
		# 			-> si alguno de los ejes es negativo tengo que mover todo para que quede en 0
		#			-> si es positivo lo dejo como esta
		position = [0 if smallestX < 0 else int(np.round(smallestX)), 0 if smallestY < 0 else int(np.round(smallestY))]

		return result, resCorners, position

	def paste( self, background, overlay, x, y):
		h, w = overlay.shape[0], overlay.shape[1]
		overlay_image = overlay[..., :3]
		mask = overlay[..., 3:] / 255.0

		print(x, y)

		background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

		return background

	def addAlphaChannel(self, image, value):
		b_channel, g_channel, r_channel = cv2.split(image)
		alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * value
		return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

	#def cropROI(self, image):
	#	upLeftCorner = (0,0)
	#	bottomRightCorner = (image.shape[1], image.shape[0])

	#	for i in range(0, image.shape[1]):
	#		for j in range(0, (image.shape[0]//4)):
				


def main():
	imagePath = 'Imagenes/'
	imageA = cv2.imread(imagePath + 'fus1.tif')
	imageB = cv2.imread(imagePath + 'fus2.tif')


	# stitch the images together to create a panorama
	stitcher = Stitcher()
	imageA = stitcher.addAlphaChannel(imageA, 255)
	imageB = stitcher.addAlphaChannel(imageB, 255)

	(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
	cv2.imwrite("resultFusarium.jpg", result)


if __name__ == '__main__':
    main()
