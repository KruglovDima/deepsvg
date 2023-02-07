import os
import csv
import pandas as pd
import pickle
import torch
import cv2
import numpy as np

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

            svg_.svg.zoom(4)
            svg_.svg.viewbox.scale(4)


            svg_.svg.save_png(png_outpath)

            img2show = cv2.imread(png_outpath)

            cv2.putText(
                img2show,
                'id: {}'.format(
                    svg_.id
                ),
                (10, 10), 
                1, 0.7, (255, 0, 0), 1
            )
            cv2.putText(
                img2show,
                'category: {} '.format(
                    svg_.category
                ),
                (10, 20), 
                1, 0.7, (255, 0, 0), 1
            )
            cv2.putText(
                img2show,
                'subcategory: {}'.format(
                    svg_.subcategory
                ),
                (10, 30), 
                1, 0.7, (255, 0, 0), 1
            )

            cv2.imwrite(png_outpath, img2show)


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
        print(type(self.svg))


        

if __name__ == '__main__':
    test_cl = SVG_from_dataset_reader(
        meta_file_path, pkl_path
    )
    test_cl.vis_tensors()

