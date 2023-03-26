import abc

class PriorityQueue(abc.ABC):
	@abc.abstractclassmethod
	def delete_min():
		pass

	@abc.abstractclassmethod
	def decrease_key(node, val) -> None:
		pass

	@abc.abstractclassmethod
	def empty() -> bool:
		pass

	@abc.abstractclassmethod
	def make_queue(nodes, dist, map) -> None:
		pass

	@abc.abstractclassmethod
	def insert(key, val) -> None:
		pass




