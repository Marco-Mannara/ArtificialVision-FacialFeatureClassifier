from skimage import feature
import numpy as np

class LBPDescriptor:
	def __init__(self, numPoints, radius, num_bins = 256, n_row = 8, n_col = 8):
		self.numPoints = numPoints
		self.radius = radius
		self.num_bins = num_bins
		self.n_row = n_row
		self.n_col = n_col

	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image,
											self.numPoints,
											self.radius,
											method="default")
		grid_lbp = self._get_grid(lbp, self.n_row, self.n_col)
		final_hist = []
		for img in grid_lbp:
			(hist, _) = np.histogram(img.ravel(),
										bins=self.num_bins,
										range=(0, 2**self.numPoints))
			# normalize the histogram
			hist = hist.astype("float")
			hist /= (hist.sum() + eps)
			final_hist.append(hist)
		# return the histogram of Local Binary Patterns
		return np.array(final_hist).ravel(), lbp

	def _get_grid(self,image, n_row=3, n_col=3):
		h = image.shape[0] // n_row
		w = image.shape[1] // n_col
		out = []
		for i in range(n_row):
			for j in range(n_col):
				out.append(image[i * h:(i + 1) * h, j * w:(j + 1) * w])
		return np.array(out)
