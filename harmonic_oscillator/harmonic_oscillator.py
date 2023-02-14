import os
import sys
import cv2
import numpy as np
import random
import math
import copy

BLACK_CL = (0, 0, 0)


class HarmonicalOscillator:
    def __init__(self, radius, start_angle, step, steps_per_it):
        # static values
        self.radius = radius
        self.start_angle = start_angle 
        self.step = step
        self.steps_per_it = steps_per_it

        self.color = (0, 0, 0)
        self.set_random_color()

        # dynamic values
        self.curr_angle = self.start_angle
        self.polyline_pts = None

    def set_random_color(self):
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # self.color = (0, 0, 0)

    def ho_step(self):
        self.curr_angle = self.curr_angle + self.step



class VisBase:
    def __init__(self, hos: list, vis_mode='normal'):
        self.hos = hos
        self.vis_mode = vis_mode
        
        self.radii = []
        self.max_radius = 0
        self.set_size()

        self.vb_height = 0
        self.vb_width = 0
        self.clear_viewbox = None
        self.temp_graph_box = None
        self.set_viewbox()

        self.static_viewbox = None
        self.set_static_viewbox()

        self.iter = 0

        self.temp_viewbox = None


    def set_size(self):
        for ho in self.hos:
            self.radii.append(ho.radius)
        self.max_radius = max(self.radii)

    def set_viewbox(self):
        self.vb_height = int(6.2 * self.max_radius)
        self.vb_width = int(2.2 * self.max_radius)

        self.clear_viewbox = np.ones([self.vb_height, self.vb_width, 3]) * 255 
        self.circle_center = (
            int(self.vb_width / 2), 
            self.vb_height - int(1.1 * self.max_radius)
        )

        self.temp_graph_box = np.ones([int(4 * self.max_radius), self.vb_width, 3]) * 255 


    def set_static_viewbox(self):
        self.static_viewbox = self.clear_viewbox
        self.border_height = 4*self.max_radius

        # circles
        for ho in self.hos:
            cv2.circle(self.static_viewbox, self.circle_center, ho.radius, ho.color, 1)

        # border between circles and graphs
        cv2.line(
            self.static_viewbox,
            pt1=(0, self.border_height), 
            pt2=(self.vb_width, self.border_height),
            color=BLACK_CL,
            thickness=1
        )

    def draw_temp(self):
        self.temp_viewbox = None
        self.temp_viewbox = copy.copy(self.static_viewbox)

        for ho in self.hos:
            # radius line
            rad_line_pt2 = (
                    int(self.circle_center[0] + ho.radius * math.sin(math.radians(ho.curr_angle))),
                    int(self.circle_center[1] + ho.radius * math.cos(math.radians(ho.curr_angle))),
                )
            cv2.line(
                self.temp_viewbox,
                pt1 = self.circle_center,
                pt2 = rad_line_pt2,
                color=ho.color,
                thickness=1
            )
            # connection line
            con_line_pt2 = (rad_line_pt2[0], self.border_height)
            cv2.line(
                self.temp_viewbox,
                pt1=rad_line_pt2,
                pt2=con_line_pt2,
                color=BLACK_CL,
                thickness=1
            )
            cv2.circle(self.temp_viewbox, center=rad_line_pt2, radius=2, color=ho.color, thickness=2)
            cv2.circle(self.temp_viewbox, center=con_line_pt2, radius=2, color=ho.color, thickness=2)

            # add point for graph to ho
            point4graph = np.array(con_line_pt2)
            if ho.polyline_pts is not None:
                move_step = 1
                bias_list = list(range(1, ho.polyline_pts.shape[0] + 1))
                bias_list = [x * move_step for x in bias_list]
                # print('bias_list::', bias_list.reverse())
                bias_arr = np.array(bias_list)
                ho.polyline_pts[:, 1] -= bias_arr
                ho.polyline_pts = np.vstack([
                    point4graph,
                    ho.polyline_pts
                ])
            else:
                ho.polyline_pts = np.expand_dims(point4graph, 0)

            if ho.polyline_pts.shape[0] >= 2:
                cv2.polylines(self.temp_graph_box, [ho.polyline_pts], isClosed=False, color=ho.color, thickness=2)

        
        self.temp_viewbox[:self.border_height, :, :] = self.temp_graph_box
        

    def step(self):
        for ho in self.hos:
            ho.ho_step()

        if self.vis_mode == 'normal':
            moved_part = self.temp_graph_box[1:, :, :]
            self.temp_graph_box *= 0
            self.temp_graph_box += 255
            self.temp_graph_box[:-1, :, :] = moved_part
        

    def run(self):
        for i in range(1000000):
            self.draw_temp()
            self.step()
            sys.stdout.write('\r{}'.format(i))
            sys.stdout.flush()

            cv2.imshow('t', self.temp_viewbox)
            cv2.waitKey(10)
            cv2.imwrite('out/ho/{:08d}.png'.format(i), self.temp_viewbox)



            
class ColorsGenerator:
    def __init__(self, start_color, index2change, step):
        self.start_color = start_color
        self.curr_color = None
        self.delta = (0, 0, 0)
        self.index2change = index2change
        self.step = step
        self.count = 0

    def get_color(self):
        if self.count == 0:
            return self.start_color
        



if __name__ == '__main__':
    # ho_s = []
    # for i in range(100):
    #     rad = random.randint(50, 300)
    #     # start_angle = random.randint(0, 360)
        
    #     start_angle_idx = random.randint(0, 3)
    #     start_angles = [45, 135, 225, 315]
    #     start_angle = start_angles[start_angle_idx]
    #     st = random.randint(1, 30)
    #     # st = 5
    #     ho = HarmonicalOscillator(radius=rad, start_angle=start_angle, step=st, steps_per_it=1)
    #     ho_s.append(ho)

    # vis = VisBase(hos=ho_s)
    # vis.run()

    rad_list = [210, 215, 220, 225, 230]
    rad_list_hos = []
    for rad in rad_list:
        start_angle_rad = 0
        step_angle_rad = 20
        for i in range(18):
            ho = HarmonicalOscillator(radius=rad, start_angle=start_angle_rad, step=5, steps_per_it=1)
            rad_list_hos.append(ho)
            start_angle_rad += step_angle_rad
    
    
    rad_list = [150, 160, 170]
    for rad in rad_list:
        start_angle_rad = 0
        step_angle_rad = 30
        for i in range(12):
            ho = HarmonicalOscillator(radius=rad, start_angle=start_angle_rad, step=7, steps_per_it=1)
            rad_list_hos.append(ho)
            start_angle_rad += step_angle_rad


    total_list = rad_list_hos
    # print(len(total_list))
    # exit()    
    vis = VisBase(hos=total_list, vis_mode='ebat')
    vis.run()
    


 