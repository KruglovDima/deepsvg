import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cairosvg

from deepsvg.svglib.svg import SVG

from deepsvg import utils
from deepsvg.svgtensor_dataset import load_dataset, SVGFinetuneDataset
from configs.deepsvg.hierarchical_ordered import Config
from deepsvg.gui.interpolate import interpolate_svg

from dataset.vis_dataset import SVG_from_dataset


class InterpolationInference:
    def __init__(self, svgs, pretrained_path, outpath):
        self.svgs = svgs 
        self.device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
        
        self.cfg = Config()
        self.cfg.model_cfg.dropout = 0.
        self.model = self.cfg.make_model().to(self.device)
        self.model.eval()

        self.pretrained_path = pretrained_path 
        self.dataset = load_dataset(self.cfg)

        self.outpath = outpath
        self.img_outpath = '{}_png'.format(self.outpath)
        self.svg_outpath = '{}_svg'.format(self.outpath)

        utils.load_model(self.pretrained_path, self.model)
        

    def finetune_model(self):
        self.finetune_dataset = SVGFinetuneDataset(self.dataset, self.svgs, frac=1.0, nb_augmentations=3500)
        self.dataloader = DataLoader(self.finetune_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False,
                        num_workers=self.cfg.loader_num_workers, collate_fn=self.cfg.collate_fn)
        
        optimizers = self.cfg.make_optimizers(self.model)
        scheduler_lrs = self.cfg.make_schedulers(optimizers, epoch_size=len(self.dataloader))
        scheduler_warmups = self.cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

        loss_fns = [l.to(self.device) for l in self.cfg.make_losses()]

        epoch = 0
        for step, data in enumerate(self.dataloader):
            self.model.train()
            model_args = [data[arg].to(self.device) for arg in self.cfg.model_args]
            labels = data["label"].to(self.device) if "label" in data else None
            params_dict, weights_dict = self.cfg.get_params(step, epoch), self.cfg.get_weights(step, epoch)
            

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(
                    zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, self.cfg.optimizer_starts), 1):
                optimizer.zero_grad()

                output = self.model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)

                loss_dict["loss"].backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                optimizer.step()
                if scheduler_lr is not None:
                    scheduler_lr.step()
                if scheduler_warmup is not None:
                    scheduler_warmup.step()

                if step % 20 == 0:
                    print(f"Step {step}: loss: {loss_dict['loss']}")

        print("Finetuning done.")

    def compute_interpolation(self):
        self.finetune_model()
        self.model.eval()

        frames_between = 50
        between_svgs = interpolate_svg(
            self.svgs[0], 
            self.svgs[1], 
            n=frames_between, 
            ease=False, 
            model=self.model, 
            viewbox_size=self.svgs[0].viewbox.size.min()
        )

        # print(type(self.svgs[:1]), type(between_svgs), type(self.svgs[1:]), type(between_svgs.reverse()), type(self.svgs[:1]))
        # full_list = self.svgs[:1] + between_svgs + self.svgs[1:] + between_svgs[::-1] + self.svgs[:1]
        full_list = self.svgs[:1] + between_svgs + self.svgs[1:]

        # for im_n, im in enumerate(full_list):
            # print(im_n, '::', im, '\nVB SIZE::   ', im.viewbox.size.min())
        # print('full_list:: ', full_list)

        for frame_n, frame in enumerate(full_list):
            print(frame_n)
            png_outpath = os.path.join(self.img_outpath, '{:06d}.png'.format(frame_n))
            svg_outpath = os.path.join(self.svg_outpath, '{:06d}.svg'.format(frame_n))

            if not os.path.exists(self.img_outpath):
                os.mkdir(self.img_outpath)
            if not os.path.exists(self.svg_outpath):
                os.mkdir(self.svg_outpath)

            # frame.save_svg(svg_outpath)
            cairosvg.svg2png(
                bytestring=frame.to_str(), 
                dpi=500,
                parent_width=int(frame.viewbox.size.min()),
                parent_height=int(frame.viewbox.size.min()),
                output_width=800,
                output_height=800,
                write_to=png_outpath, 
                background_color='white'
            )

            cairosvg.svg2svg(
                bytestring=frame.to_str(), 
                dpi=500,
                parent_width=int(frame.viewbox.size.min()),
                parent_height=int(frame.viewbox.size.min()),
                output_width=24,
                output_height=24,
                write_to=svg_outpath, 
                background_color='white'
            )
        


if __name__ == '__main__':
    # image outpath
    img_outpath = '/home/ilpech/repositories/thirdparty/deepsvg/out/interpolation_001'
    pretrained_path = './pretrained/hierarchical_ordered.pth.tar'
    
    # icon8 dataset
    pkl_path = '/home/ilpech/repositories/thirdparty/deepsvg_/dataset/icons_tensor'
    meta_file_path = '/home/ilpech/repositories/thirdparty/deepsvg_/dataset/icons_meta.csv'
    meta_file = pd.read_csv(meta_file_path, header=0)

    # load pattern svg from .svg format
    # pattern_svg_path = '/home/ilpech/repositories/thirdparty/deepsvg/dataset/patterns/sensesay_logo_red.svg'
    # pattern_svg_path = '/home/ilpech/repositories/thirdparty/deepsvg/dataset/patterns/sensesay_logo_simple.svg'
    pattern_svg_path = '/home/ilpech/repositories/thirdparty/deepsvg/dataset/patterns/Patterns.svg'
    # pattern_svg_path = '/home/ilpech/repositories/thirdparty/deepsvg/dataset/patterns/test_rect.svg'
    # pattern_svg = SVG.load_svg(pattern_svg_path)

    # pattern_svg.fill_()
    # pattern_svg.canonicalize()
    # pattern_svg.normalize()
    # pattern_svg = pattern_svg.simplify_heuristic()
    # exit()

    # cairosvg.svg2svg(
    #     bytestring=pattern_svg.to_str(), 
    #     dpi=500,
    #     parent_width=int(pattern_svg.viewbox.size.min()),
    #     parent_height=int(pattern_svg.viewbox.size.min()),
    #     output_width=800,
    #     output_height=800,
    #     write_to='out/trash/patterns.svg', 
    #     background_color='white'
    # )

    # cairosvg.svg2png(
    #     bytestring=pattern_svg.to_str(), 
    #     dpi=500,
    #     parent_width=int(pattern_svg.viewbox.size.min()),
    #     parent_height=int(pattern_svg.viewbox.size.min()),
    #     output_width=800,
    #     output_height=800,
    #     write_to='out/trash/patterns.png', 
    #     background_color='white'
    # )


    # load sample from icon8 dataset
    # weather_id = 4521 #weather cloud
    # dataset_svg_sample = SVG_from_dataset(weather_id, meta_file, pkl_path)

    # dataset_svg = dataset_svg_sample.svg
    # dataset_svg.fill_()
    # dataset_svg.normalize()
    # dataset_svg.canonicalize()
    # dataset_svg = dataset_svg.simplify_heuristic()

    # frames2interpolate = [pattern_svg, dataset_svg]


    # interpolate res from interpolation
    pattern_svg_path_1 = 'out/interpolation_001_svg/000000.svg'
    pattern_svg_1 = SVG.load_svg(pattern_svg_path_1)

    pattern_svg_1.fill_()
    pattern_svg_1.canonicalize()
    pattern_svg_1.normalize()
    pattern_svg_1 = pattern_svg_1.simplify_heuristic()
    
    pattern_svg_path_2 = 'out/interpolation_001_svg/000001.svg'
    pattern_svg_2 = SVG.load_svg(pattern_svg_path_2)

    pattern_svg_2.fill_()
    pattern_svg_2.canonicalize()
    pattern_svg_2.normalize()
    pattern_svg_2 = pattern_svg_2.simplify_heuristic()

    frames2interpolate = [pattern_svg_1, pattern_svg_2]


    # create inference calculator
    inference = InterpolationInference(
        frames2interpolate,
        pretrained_path,
        img_outpath
    )

    inference.compute_interpolation()








