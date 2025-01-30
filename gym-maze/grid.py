import random


class Grid:
	def __init__(self, x, cell_type=None):
		if not cell_type: cell_type = type(x[0][0])
		self.grid = [[cell_type(a) for a in list(line)] for line in x]
		self.height = len(self.grid)
		self.width = len(self.grid[0])
		self.h = self.height
		self.w = self.width
		self.area = self.height * self.width


	def __getitem__(self, x):
		return self.grid[x]


	def __iter__(self):
		for r in range(self.height):
			for c in range(self.width):
				yield self.grid[r][c]


	def __str__(self):
		return '\n'.join([''.join(map(str,line)) for line in self.grid])


	def get_neighbours(self, point, rand=False):
		(r, c) = point
		neighbour_cells = [(r-2, c), (r+2, c), (r, c-2), (r, c+2)]

		if rand:
			random.shuffle(neighbour_cells)

		for (r, c) in neighbour_cells:
			if self.valid(r, c):
				yield (r, c)


	def all_points(self):
		for r in range(self.height):
			for c in range(self.width):
				yield (r, c)


	def count(self, symbol):
		return sum([line.count(symbol) for line in self.grid])


	def valid(self, r, c):
		return (0 <= r < self.height) and (0 <= c < self.width)


	def get(self, point):
		(r, c) = point
		return self.grid[r][c]


	def set(self, point, symbol):
		(r, c) = point
		self.grid[r][c] = symbol