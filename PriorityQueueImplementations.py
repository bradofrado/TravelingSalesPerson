from PriorityQueue import PriorityQueue


class ArrayPriorityQueue(PriorityQueue):
	def __init__(self):
		self.dist = {}
		self.count = 0
		self.keys = set()
		self.map = None
		pass

	# Time: The complexity is that of the for loop, O(V), + keys.remove (O(V)) + dist.pop (O(1))
	#       So overall O(V)
	# Space: The space complexity is at that of keys and dist, which is O(V)
	def delete_min(self):
		minIndex = -1
		keys = self.keys

		#The time complexity here is looping through each node in keys, so O(V)
		for key in keys:
			if minIndex == -1 or self.dist[key] < self.dist[minIndex]:
				minIndex = key

		if minIndex == -1:
			return None
		
		# The time complexity of remove is looping through each node till it can be removed, or O(V)
		self.keys.remove(minIndex)

		#Popping off a dictionary is just a simple lookup, so O(1)
		self.dist.pop(minIndex)
		return minIndex

	# Time: Doing a simple lookup on a dictionary is O(1)
	# Space: We need the self.dict dictionary which is O(V)
	def decrease_key(self, key, val):
		self.dist[key] = val

	# Time: len(self.keys) is a simple property value, so O(1)
	# Space: We need self.keys for this function, so O(V)
	def empty(self):
		return len(self.keys) == 0

	# Time: This function loops through all of the keys, which is all the nodes, so O(V)
	# Space: Since we need keys and dist which is all the nodes, it is O(V)
	def make_queue(self, keys, dist, map = lambda x: x):
		self.map = map
		for i in range(len(keys)):
			self.insert(keys[i], dist[i])

  # Time: Adding a value at the back of a list keys is O(1) time. Same for the dictionary dist
	# Space: We need all of dist and keys for this, so O(V)
	def insert(self, key, val):
		self.dist[key] = val
		key = self.map(key) if self.map != None else key
		self.keys.add(key)

class HeapPriorityQueue(PriorityQueue):
	def __init__(self):
		self.heap = {}
		self.pointer = {}
		self.count = 0
		pass

	# Time: All of these operations are O(1) time except pDown, which is O(log(V)), so overall O(log(V))
	# Space: we need the whole heap and pointer array for this, so O(V)
	def delete_min(self):
		min = self.heap[0][0]
		self.pointer[self.heap[0][0]] = None
		self.heap[0] = None
		self.switch(0, self.count - 1)
		self.count -= 1
		self.pDown(0)
		return min

	# Time: The only time crucher in this is pUp, which is O(log(V)), so overall O(log(V))
	# Space: Since we need pointer and heap arrays, it is O(V)
	def decrease_key(self, key, val):
		index = self.pointer[key]
		if index == None:
			return

		self.heap[index][1] = val
		self.pUp(index)

	# Time: Pretty self explanitory how this is O(1)!
	# Space: O(1) also, one variable
	def empty(self):
		return self.count == 0

	# Time: looping through the keys and then performing insert is O(V*insert) or O(V*log(V))
	# Space: We only need space for keys and dist, which is all the nodes so O(V)
	def make_queue(self, keys, dist, map = lambda x: x):
		keys = mapKeys(keys, map)
		for i in range(len(keys)):
			key = keys[i]
			self.insert(key, dist[key])

	# Time: This complexity comes from the function pUp, which is O(logV)
	# Space: We need space for heap and pointer, so O(V)
	def insert(self, key, val):
		i = self.count
		self.heap[i] = [key, val]
		self.pointer[key] = i
		
		self.pUp(i)
		self.count += 1

	# Time: This function is a recursive function that goes, from the bottom potentially, up
	#       The binary tree. Since a binary tree has logN depth and that is the most this function
	#       well ever traverse, the complexity is O(logV)
	# Space: We need space for all the nodes in heap, so O(V)
	def pUp(self, index):
		parentKey = self.heap[index // 2] if index > 0 else None
		if parentKey == None:
			return

		curr = self.heap[index]

		if curr[1] < parentKey[1]:
			self.switch(index, index // 2)
			self.pUp(index // 2)

	# Time: Similar to pUp, this function will recurse from the top to the bottom of the tree
	#       at a depth of logV, so O(logV)
	# Space: We need space for the heap of nodes, so O(V)
	def pDown(self, index):
		if index * 2 + 1 >= self.count:
			return

		left = self.heap[index * 2 + 1]
		right = self.heap[index * 2 + 2] if self.count > index * 2 + 2 else None

		switchIndex = -1
		if right == None or left[1] < right[1]:
			switchIndex = index * 2 + 1
		elif right[1] < left[1]:
			switchIndex = index * 2 + 2	

		if switchIndex != -1 and self.heap[index][1] > self.heap[switchIndex][1]:
			self.switch(index, switchIndex)
			self.pDown(switchIndex)

	# Time: This is a O(1) operation function because the lookups are O(1), and that's all this function is
	# Space: We need space for the heap, so O(V)
	def switch(self, src, dest):
		srcKey = self.heap[src]
		destKey = self.heap[dest]
		self.heap[src] = destKey
		self.heap[dest] = srcKey
		if srcKey != None:
			self.pointer[srcKey[0]] = dest
		if destKey != None:
			self.pointer[destKey[0]] = src

# Time: This function is a simple append for all the keys, so O(n). I use n instead of V here because
#      This function is not dependent on graphs or any V's
# Space: The space is that of keys, or O(n)
def mapKeys(keys, map):
	out = []
	for key in keys:
		out.append(map(key))

	return out