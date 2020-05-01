class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a lis'objects't of centroids
		# using the current centroid
		self.objectID = objectID
		self.firstpos=0
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False