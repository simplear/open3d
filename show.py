# -*- Coding: utf-8 -*-

# show pointcloud
# refer
# - http://www.open3d.org/docs/release/tutorial/Basic/visualization.html

import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("./data/xw_c.ply")
o3d.visualization.draw_geometries([pcd])