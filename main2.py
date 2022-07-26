import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import qdarkstyle
import qdarkgraystyle
from time import time
from options.test_options import TestOptions
from ui.ui import Ui_Form
import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import cv2
from ui.mouse_event import GraphicsScene
from ui.GT_mouse_event import GTScene
from utils import Build_model
import pickle
from sklearn.manifold import TSNE
from ui.ui import transfer_real_to_slide, invert_slide_to_real, light_transfer_real_to_slide, \
    light_invert_slide_to_real, attr_degree_list
import torch
from module.flow import cnf
import os
import tensorflow as tf
from ui.real_time_attr_thread import RealTimeAttrThread
from ui.real_time_light_thread import RealTimeLightThread

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Ex:
    def lightn(self, lid):
        return torch.from_numpy(self.raw_lights[lid]).type(torch.FloatTensor).cuda()
    def init(self, opt):
        self.at_intial_point = False
        self.keep_indexes = [2, 5]
        self.keep_indexes = np.array(self.keep_indexes).astype(np.int)
        self.zero_padding = torch.zeros(1, 18, 1).cuda()
        self.attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
        self.lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']
        self.model = Build_model(opt)
        #self.w_avg = self.model.Gs.get_var('dlatent_avg')
        self.prior = cnf(512, '512-512-512-512-512', 17, 1) #seq model to convert 17 (9lights + 8attr) to 1,18,512 (w+ space)
        self.prior.load_state_dict(torch.load('flow_weight/modellarge10k.pt'))
        self.prior.eval()
        self.raw_w = pickle.load(open("data/sg2latents.pickle", "rb")) #presets of w+ space (idx by keep_indexes)
        self.raw_TSNE = np.load('data/TSNE.npy')
        self.raw_attr = np.load('data/attributes.npy')
        self.raw_lights = np.load('data/light.npy')
        self.all_w = np.array(self.raw_w['Latent'])[self.keep_indexes]
        self.all_attr = self.raw_attr[self.keep_indexes]
        self.all_lights = self.raw_lights[self.keep_indexes]
        self.pre_lighting = [
            self.lightn(8), self.lightn(33), self.lightn(641), self.lightn(547), self.lightn(28), self.lightn(34)]

    def wcurrent_rev(self):
        self.w_current = self.rev[0].detach().cpu().numpy()

    def qa_fws(self):        
        self.q_array = torch.from_numpy(self.w_current).cuda().clone().detach()
        self.fws = self.prior(self.q_array, self.final_array_target, self.zero_padding)

    def update_GT_scene_image(self, pickedImageIndex):
        self.light_current_list = [0 for i in range(len(self.lighting_order))]
        self.attr_current = self.all_attr[pickedImageIndex].copy()
        self.attr_current_list = [self.attr_current[i][0] for i in range(len(self.attr_order))]
        self.array_source = torch.from_numpy(self.attr_current).type(torch.FloatTensor).cuda()
        self.array_light = torch.from_numpy(self.all_lights[pickedImageIndex].copy()).type(torch.FloatTensor).cuda()
        self.final_array_target = torch.cat([self.array_light, self.array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        self.pre_lighting_distance = [self.pre_lighting[i] - self.array_light for i in range(len(self.lighting_order))]
        self.w_current = self.all_w[pickedImageIndex].copy()
        self.qa_fws()

    def predict_w_plus(self):
        self.rev = self.prior(self.fws[0], self.final_array_target, self.zero_padding, True)

    def real_time_lighting(self, light_index, raw_slide_value):
        #real_value = light_invert_slide_to_real(self.lighting_order[light_index], raw_slide_value)
        real_value = raw_slide_value
        self.light_current_list[light_index] = real_value
        lighting_final = self.array_light.clone().detach()
        for i in range(len(self.lighting_order)):
            lighting_final += self.light_current_list[i] * self.pre_lighting_distance[i]
        self.final_array_target[:, :9] = lighting_final

        self.predict_w_plus()
        self.rev[0][0][0:7] = self.q_array[0][0:7]
        self.rev[0][0][12:18] = self.q_array[0][12:18]
        self.wcurrent_rev()
        self.qa_fws()

    def real_time_editing(self, attr_index, raw_slide_value):
        #real_value = invert_slide_to_real(self.attr_order[attr_index], raw_slide_value)
        real_value = raw_slide_value
        attr_change = real_value - self.attr_current_list[attr_index]
        attr_final = attr_degree_list[attr_index] * attr_change + self.attr_current_list[attr_index]
        self.final_array_target[0, attr_index + 9, 0, 0] = attr_final

        self.predict_w_plus()
        if attr_index == 0:
            self.rev[0][0][8:] = self.q_array[0][8:]
        elif attr_index == 1:
            self.rev[0][0][:2] = self.q_array[0][:2]
            self.rev[0][0][4:] = self.q_array[0][4:]
        self.wcurrent_rev()
        self.qa_fws()

    def QISave(self, imgfile):
        self.GAN_image = self.model.generate_im_from_w_space(self.w_current)[0]
        qim = QImage(self.GAN_image.data, self.GAN_image.shape[1], self.GAN_image.shape[0], self.GAN_image.strides[0],
                     QImage.Format_RGB888)
        qim.save(imgfile, 'JPG')

if __name__ == '__main__':
    print('start')
    e=Ex()
    opt = TestOptions().parse()
    e.init(opt)
    e.update_GT_scene_image(1)
    e.QISave('img.JPG')
    e.real_time_lighting(1, 1)
    e.QISave('img2.JPG')
    e.real_time_editing(1, 1)
    e.QISave('img3.JPG')
