from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import PIL.Image as Image
import numpy as np
from config import settings
import torch


class remote_val():

    """voc dataset."""

    def __init__(self, args, transform=None, k_shot=1):
        self.num_classes = 15
        self.group = args.group
        self.num_folds = args.num_folds
        #self.binary_map_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'list/val') #val
        self.data_list_dir = '/disk2/caoqinglong/remote_sensing/iSAID_patches/val/val_list'
        self.img_dir = os.path.join(settings.DATA_DIR, 'remote_sensing/iSAID_patches/val/', 'images/')
        self.mask_dir = os.path.join(settings.DATA_DIR,'remote_sensing/iSAID_patches/val/', 'semantic_png/')
        #self.binary_mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/val/')
        self.list_splite = self.get_total_list()
        self.list_splite_len = len(self.list_splite)
        self.list_class = self.get_class_list()
        self.transform = transform
        self.count = 0
        self.random_generator = random.Random()
        self.random_generator.seed(1385) #1385
        self.k_shot = k_shot

    def get_total_list(self):
        new_exist_class_list = []
        f = open(os.path.join(self.data_list_dir, 'split%1d_val.txt' % (self.group)))
        while True:
            item = f.readline()
            if item == '':
                break
            img_name = item.strip(item[-4:]).strip('_instance_color_RGB.png')
            cat = int(item[-3:])-1
            new_exist_class_list.append([img_name, cat])
        print("Total images are : ", len(new_exist_class_list))
        return new_exist_class_list

    def get_class_list(self):
        list_class = {}
        for i in range(self.num_classes):
            list_class[i] = []
        for name, class_ in self.list_splite:
            if class_ < 0:
                print(name)
            list_class[class_].append(name)

        return list_class

    def read_img(self, name):
        # a = name[:2]
        # if a=='wh':
        #     path = self.img_dir + name + '.jpg'
        # else:
        #     path = self.img_dir + name + '.tif'
        path = self.img_dir + name +'.png'
        img = Image.open(path)

        return img

    def read_mask(self, name, category):
        path = self.mask_dir + name + '_instance_color_RGB.png'
        mask = cv2.imread(path)

        mask[mask==255] = 0

        mask[mask!=category+1] = 0
        mask[mask==category+1] = 1

        return mask[:,:,0].astype(np.float32)
    '''
    def read_binary_mask(self, name, category):
        path = self.binary_mask_dir +str(category+1)+'/'+ name + '.png'
        mask = cv2.imread(path)/255

        #mask[mask!=category+1] = 0
        #mask[mask==category+1] = 1

        return mask[:,:,0].astype(np.float32)
    '''
    def load_frame(self, support_name, query_name, class_):
        support_img = self.read_img(support_name)
        query_img = self.read_img(query_name)
        support_mask = self.read_mask(support_name, class_)
        query_mask = self.read_mask(query_name, class_)

        #support_mask = self.read_binary_mask(support_name, class_)
        #query_mask = self.read_binary_mask(query_name, class_)

        return query_img, query_mask, support_img, support_mask, class_

    def load_frame_k_shot(self, support_name_list, query_name, class_):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name, class_)

        support_img_list = []
        support_mask_list = []

        for support_name in support_name_list:
            support_img = self.read_img(support_name)
            support_mask = self.read_mask(support_name, class_)
            support_img_list.append(support_img)
            support_mask_list.append(support_mask)

        return query_img, query_mask, support_img_list, support_mask_list


    def get_1_shot(self, idx):
        if self.count >= self.list_splite_len:
            self.random_generator.shuffle(self.list_splite)
            self.count = 0
        query_name, class_ = self.list_splite[self.count]
        # query_name = 'P0791_412_668_0_256'
        # class_ = 7
        #
        
        while True:  # random sample a support data
            support_img_list = self.list_class[class_]
            support_name = support_img_list[self.random_generator.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)
 
        # while True:
        #     if self.count >= self.list_splite_len:
        #         self.random_generator.shuffle(self.list_splite)
        #         self.count = 0
        #     query_name, class_ = self.list_splite[self.count]
        #     # query_name = '2008_006641'
        #     # class_ = 4
        #     while True:  # random sample a support data
        #         support_img_list = self.list_class[class_]
        #         support_name = support_img_list[self.random_generator.randint(0, len(support_img_list) - 1)]
        #         if support_name != query_name:
        #             break
        #     query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)
        #     sum1 = query_mask.sum()
        #     sum2 = support_mask.sum()
        #     if sum1 <= 64*64:
        #         break
        #     # if sum1 >= 32*32 and sum2 >= 32*32:
        #     #     break
        #     else:
        #         self.count = self.count + 1


        size = query_mask.shape

        if self.transform is not None:
            query_img, query_mask = self.transform(query_img, query_mask)
            support_img, support_mask = self.transform(support_img, support_mask)

        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_, size,query_name

    def get_k_shot(self, idx):
        if self.count >= self.list_splite_len:
            self.random_generator.shuffle(self.list_splite)
            self.count = 0

        # while True:
        #     if self.count >= self.list_splite_len:
        #         self.random_generator.shuffle(self.list_splite)
        #         self.count = 0
        #     query_name, class_ = self.list_splite[self.count]
            
            
            #   # random sample 5 support data
            # support_set_list = self.list_class[class_]
            # support_choice_list = support_set_list.copy()
            # support_choice_list.remove(query_name)
            # support_name_list = self.random_generator.sample(support_choice_list, self.k_shot)
            # query_img, query_mask, support_img_list, support_mask_list = self.load_frame_k_shot(support_name_list, query_name, class_)
            # sum1 = query_mask.sum()
            # sum2_0 = support_mask_list[0].sum()
            # sum2_1 = support_mask_list[1].sum()
            # sum2_2 = support_mask_list[2].sum()
            # sum2_3 = support_mask_list[3].sum()
            # sum2_4 = support_mask_list[4].sum()
            # k =0
            # if sum1 >= 32 * 32 :
            #     k=k+1
            # if sum2_0 >= 32 * 32:
            #     k = k+1
            # if sum2_1 >= 32 * 32:
            #     k = k+1
            # if sum2_2 >= 32 * 32:
            #     k = k+1
            # if sum2_3 >= 32 * 32:
            #     k = k+1
            # if sum2_4 >= 32 * 32:
            #     k = k+1
            #
            # if k==6:
            #     break
            # else:
            #     self.count = self.count + 1
                
        query_name, class_ = self.list_splite[self.count]
        support_set_list = self.list_class[class_]
        support_choice_list = support_set_list.copy()
        support_choice_list.remove(query_name)
        support_name_list = self.random_generator.sample(support_choice_list, self.k_shot)
        query_img, query_mask, support_img_list, support_mask_list = self.load_frame_k_shot(support_name_list, query_name, class_)

        size = query_mask.shape

        if self.transform is not None:
            query_img, query_mask = self.transform(query_img, query_mask)
            for i in range(len(support_mask_list)):
                support_temp_img = support_img_list[i]
                support_temp_mask = support_mask_list[i]
                support_temp_img, support_temp_mask = self.transform(support_temp_img, support_temp_mask)
                support_temp_img = support_temp_img.unsqueeze(dim=0)
                support_temp_mask = support_temp_mask.unsqueeze(dim=0)
                if i ==0:
                    support_img = support_temp_img
                    support_mask = support_temp_mask
                else:
                    support_img = torch.cat([support_img, support_temp_img], dim=0)
                    support_mask = torch.cat([support_mask, support_temp_mask], dim=0)


        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_, size,query_name
    
    def num_to_label(self,class_num):
        label_set =  [1,2,3,4,5,6,7,8,9,10]
        label_one_hot = torch.zeros(10).long()
        if self.group == 0:
            # label_set = [5,6,7,8,9,10,11,12,13,14]
            class_num =  class_num-4
            assert class_num in label_set
            label_one_hot[class_num-1]=1
            
        if self.group == 1:
            # label_set = [0,1,2,3,4,10,11,12,13,14]
            if class_num in [0,1,2,3,4]:
                class_num = class_num+1
            else:
                class_num = class_num-4
            assert class_num in label_set
            label_one_hot[class_num-1]=1
            
        if self.group == 2:
            # label_set = [0,1,2,3,4,5,6,7,8,9]
            class_num =  class_num+1
            assert class_num in label_set
            label_one_hot[class_num-1]=1

        return label_one_hot,class_num-1
    
    def __len__(self):
        # return len(self.image_list)
        return 1000


    def __getitem__(self, idx):
        if self.k_shot==1:
            query_img, query_mask, support_img, support_mask, class_, size,query_name  = self.get_1_shot(idx)# , size
        else:
            query_img, query_mask, support_img, support_mask, class_, size,query_name = self.get_k_shot(idx)  # , size
          
        return query_img, query_mask, support_img, support_mask, class_,size


