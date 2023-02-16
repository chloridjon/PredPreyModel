import __parameters as p
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

r = np.array([[0,0],[1,8],[2,4],[-3,12]])
V = Voronoi(r)
print(type(V))

indices = V.point_region
regions = V.regions
point_regions = [regions[i] for i in indices]

print(type(point_regions))