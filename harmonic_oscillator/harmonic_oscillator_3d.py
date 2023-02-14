import os
import random
import numpy as np
import math

import open3d as o3d

class HarmonicOscillator3d:
    def __init__(self, radius, start_deg, degs_per_step):
        self.radius = radius
        self.start_deg = start_deg
        self.degs_per_step = degs_per_step
        self.color = np.array([
            random.random(),
            random.random(),
            random.random()
        ])
        self.curr_deg = start_deg

    def step(self):
        self.curr_deg += self.degs_per_step



class HOVisualizer:
    def __init__(self, hos):
        self.hos = hos
        self.points = None
        self.colors = None

        self.z = 0

        self.radii = []
        self.max_rad = 0
        self.circle_center = None
        self.calculate_circle_center()

    def calculate_circle_center(self):
        for ho in self.hos:
            self.radii.append(ho.radius)

        self.max_rad = max(self.radii)
        self.circle_center = [self.max_rad, self.max_rad]


    def calculate_points(self, z_bias, ho_point_num):
        for i in range(ho_point_num):
            for ho in self.hos:
                ho_x = ho.radius * math.cos(math.radians(ho.curr_deg)) + self.circle_center[0]
                ho_y = ho.radius * math.sin(math.radians(ho.curr_deg)) + self.circle_center[1]

                ho_xyz = np.array([ho_x, ho_y, self.z])
                if self.points is not None:
                    self.points = np.vstack([
                        self.points,
                        ho_xyz
                    ])

                else:
                    self.points = ho_xyz

                if self.colors is not None:
                    self.colors = np.vstack([
                        self.colors,
                        ho.color
                    ])
                else:
                    self.colors = ho.color

                ho.step()

            self.z += z_bias
        
    def visualize(self):
        if self.points is not None:
            pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.colors = o3d.utility.Vector3dVector(self.colors)

            o3d.visualization.draw_geometries([pcd])

            

if __name__ == '__main__':
    # degs_list = [0, 45, 90, 135, 180, 225, 270, 315]
    degs_list = [0, 7, 14, 28, 54, 108, 216]
    rads_list = [100, 200, 300, 400, 500]
    hos_list = []
    st = 1
    for rad in rads_list:
        for deg in degs_list:
            ho = HarmonicOscillator3d(rad, deg, st)
            hos_list.append(ho)
        st+=1

    # for deg in degs_list:
    #     ho = HarmonicOscillator3d(400, deg, 1)
    #     hos_list.append(ho)
    
    # for deg in degs_list:
    #     ho = HarmonicOscillator3d(300, deg, 1)
    #     hos_list.append(ho)

    visualizer = HOVisualizer(hos_list)
    visualizer.calculate_points(5, 1000)
    visualizer.visualize()
    # print(visualizer.points)


                




    




    
