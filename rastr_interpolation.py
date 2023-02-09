import os
import cv2

data_dir = '/home/ilpech/repositories/thirdparty/deepsvg/out/trash'

class RastrInterpolator:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.original_imgs_names = []
        for r, ds, fs in os.walk(self.data_dir):
            for f in fs:
                self.original_imgs_names.append(f)
        self.original_imgs_names = sorted(self.original_imgs_names)

        self.original_imgs = []
        for original_img_name in self.original_imgs_names:
            path = os.path.join(self.data_dir, original_img_name)
            img = cv2.imread(path)
            self.original_imgs.append(img)

        self.cur_frame_id = 0
        self.saved_count = 0

        

    def rastr_interpolation(self):
        for fr_id, frame in enumerate(self.original_imgs):
            print('FRAME_ID:: ', fr_id)
            if self.cur_frame_id != len(self.original_imgs):
                mean_img = cv2.addWeighted(
                    self.original_imgs[self.cur_frame_id], 0.8, 
                    self.original_imgs[self.cur_frame_id + 1], 0.8, 
                    1
                )
                if self.cur_frame_id == 0:
                    cv2.imwrite(
                        os.path.join(self.out_dir,'{:08d}.png'.format(self.saved_count)), 
                        self.original_imgs[self.cur_frame_id]
                    )
                    self.saved_count += 1

                cv2.imwrite(
                    os.path.join(
                        self.out_dir,
                        '{:08d}.png'.format(self.saved_count)
                    ), 
                    mean_img
                )
                self.saved_count += 1

                cv2.imwrite(
                    os.path.join(
                        self.out_dir,
                        '{:08d}.png'.format(self.saved_count)
                    ), 
                    self.original_imgs[self.cur_frame_id + 1]
                )
                self.saved_count += 1
                self.cur_frame_id += 1

if __name__ == '__main__':
    data_dir = '/home/ilpech/repositories/thirdparty/deepsvg/out/trash_interpolated_4'
    out_dir = '/home/ilpech/repositories/thirdparty/deepsvg/out/trash_interpolated_5'

    interpolator = RastrInterpolator(data_dir, out_dir)
    interpolator.rastr_interpolation()



        

