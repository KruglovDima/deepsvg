import os
import csv
import pandas as pd
import pickle
import torch
import cv2
import numpy as np
import cairosvg

from deepsvg.gui.interpolate import decode
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox

# pkl_path = 'dataset/icons_tensor_small'
# meta_file_path = 'dataset/icons_meta_small.csv'
# out_dir = 'out/icon8_png'
pkl_path = '/home/ilpech/repositories/thirdparty/deepsvg_/dataset/icons_tensor'
meta_file_path = '/home/ilpech/repositories/thirdparty/deepsvg_/dataset/icons_meta.csv'
out_dir = 'out/icon8_png'



class SVG_from_dataset_reader:
    def __init__(self, meta_file_path, pkl_dir_path):
        self.meta_file = pd.read_csv(meta_file_path, header=0)
        self.pkl_dir_path = pkl_dir_path
        self.out_dir = out_dir

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)

        self.categories = {}

        # create ids list
        try:
        # if 'id' in self.meta_file.head(0).columns.to_list():
            self.ids_list = self.meta_file['id'].to_list()
        except:
            print('No ids in meta file.')

    def vis_tensors(self):
        # converts each tensors id from pkl to svg
        for id in self.ids_list:
            svg_ = SVG_from_dataset(id, self.meta_file, self.pkl_dir_path)
            png_outpath = os.path.join(self.out_dir, '{:06d}.png'.format(id))
            # png_outpath = 'out/test_1.png'
            # svg_.svg.zoom(4)
            # svg_.svg.viewbox.scale(4)
            
            # create category dict
            print(svg_.category, svg_.subcategory)
            if svg_.category not in self.categories:
                self.categories[svg_.category] = []
            
            if svg_.subcategory not in self.categories[svg_.category]:
                self.categories[svg_.category].append(svg_.subcategory)

            # print(self.categories)

            cairosvg.svg2png(
                bytestring=svg_.svg.to_str(), 
                dpi=500,
                parent_width=200,
                parent_height=200,
                output_width=800,
                output_height=800,
                write_to=png_outpath, 
                background_color='white'

            )
            # svg_.svg.save_png(png_outpath)

            img2show = cv2.imread(png_outpath)

            # img2show = cv2.dilate(
            #     img2show, 
            #     kernel=np.ones((1, 1), 'uint8'),
            #     iterations=5
            # )

            fontface = 2
            fontscale = 1
            text_color = (0, 0, 0)
            thickness = 1

            
            cv2.putText(
                img2show,
                'id: {}'.format(
                    svg_.id
                ),
                (10, 30), 
                fontface, fontscale, text_color, thickness
            )
            cv2.putText(
                img2show,
                'category: {} '.format(
                    svg_.category
                ),
                (10, 60), 
                fontface, fontscale, text_color, thickness
            )
            cv2.putText(
                img2show,
                'subcategory: {}'.format(
                    svg_.subcategory
                ),
                (10, 90), 
                fontface, fontscale, text_color, thickness
            )

            cv2.imwrite(png_outpath, img2show)

    def save_category_dict(self):
        text_path = '{}_categories.txt'
        with open(text_path, 'w') as f:
            for key in self.categories.keys:
                info_line = '{}:: {}'.format(key, self.categories[key])
                f.write(info_line)
                f.write('\n')
            


class SVG_from_dataset:
    def __init__(self, id: int, meta_file: pd.core.frame.DataFrame(), pkl_dir_path):
        self.id = id
        self.meta_file = meta_file

        self.category = self.meta_file.loc[self.meta_file['id'] == self.id]['category'].to_string(index=False)
        self.subcategory = self.meta_file.loc[self.meta_file['id'] == \
                                         self.id]['subcategory'].to_string(index=False).replace('-', '_')

        self.pkl_dir_path = pkl_dir_path

        self.pkl_path = os.path.join(
            pkl_dir_path, '{}.pkl'.format(self.id)
        )

        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.tensors = data['tensors']
        self.fillings = data['fillings']

        self.PAD_VAL = -1
        self.MAX_SEQ_LEN = 30
        self.MAX_NUM_GROUPS = 8

        self.svg = None
        # convert tensor to svg:
        self.tensor2svg()

    def tensor2svg(self):
        t_sep = self.tensors[0]
        tensor_pred = SVG.from_tensors(t_sep, viewbox=Bbox(200))
        self.svg = tensor_pred


if __name__ == '__main__':
    test_cl = SVG_from_dataset_reader(
        meta_file_path, pkl_path
    )
    test_cl.vis_tensors()
    test_cl.save_category_dict()

