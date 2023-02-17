from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import PIL.Image as Image
import numpy as np
from config import settings
import torch
#random.seed(1385)

class remote_train():

    """Face Landmarks dataset."""

    def __init__(self, args, transform=None, k_shot=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 15
        self.group = args.group
        self.num_folds = args.num_folds
        #self.binary_map_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train') #val
        # self.data_list_dir = os.path.join('data_list/train')
        self.data_list_dir = '/disk2/caoqinglong/remote_sensing/iSAID_patches/train/train_list'
        self.img_dir = os.path.join(settings.DATA_DIR, 'remote_sensing/iSAID_patches/train/', 'images/')
        self.mask_dir = os.path.join(settings.DATA_DIR, 'remote_sensing/iSAID_patches/train/', 'semantic_png/')
        #self.binary_mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train/')

        self.train_id_list = self.get_train_id_list()
        self.list_splite = self.get_total_list()
        self.list_splite_len = len(self.list_splite)
        self.list_class = self.get_class_list()
        self.k_shot = k_shot
        self.transform = transform
        self.count = 0
        self.random_generator = random.Random()
        self.len = args.max_steps *args.batch_size *2
        #self.random_generator.shuffle(self.list_splite)
        #self.random_generator.seed(1385)
        
    def get_train_id_list(self):
        if self.group == 0:
            val_set = [0,1,2,3,4]
        if self.group == 1:
            val_set = [5,6,7,8,9]
        if self.group == 2:
            val_set = [10,11,12,13,14]
        train_set = [x for x in range(self.num_classes) if x not in val_set]

        return train_set

    def get_total_list(self):
        new_exist_class_list = []

        fold_list = [0, 1, 2]
        fold_list.remove(self.group)

        # f = open(os.path.join(self.data_list_dir, 'split%1d_train.txt' % (self.group)))
        for fold in fold_list:
            f = open(os.path.join(self.data_list_dir, 'split%1d_train.txt' % (fold)))
            while True:
                item = f.readline()
                if item == '':
                    break
                
                cat = int(item[-3:]) -1

                img_name = item.strip(item[-4:]).strip('_instance_color_RGB.png')

                new_exist_class_list.append([img_name, cat])

        print("Total images are : ", len(new_exist_class_list))
        # if need filter
        # new_exist_class_list = self.filte_multi_class(new_exist_class_list)
        return new_exist_class_list

    def filte_multi_class(self, exist_class_list):

        new_exist_class_list = []
        for name, class_ in exist_class_list:

            mask_path = self.mask_dir + name + '_instance_color_RGB.png'
            # print(mask_path)
            mask = cv2.imread(mask_path)
        
            labels = np.unique(mask[:,:,0])
            labels = [label - 1 for label in labels if label != 255 and label != 0]
            if set(labels).issubset(self.train_id_list):
                new_exist_class_list.append([name, class_])
            # if set(labels).issubset(self.train_id_list):
            # new_exist_class_list.append([name, class_])
        print("Total images after filted are : ", len(new_exist_class_list))
        return new_exist_class_list


    def get_class_list(self):
        list_class = {}
        for i in range(self.num_classes):
            list_class[i] = []
        for name, class_ in self.list_splite:
            list_class[class_].append(name)

        return list_class

    def read_img(self, name):
        # a = name[:2]
        # if a=='wh':
        #     path = self.img_dir + name + '.jpg'
        # else:
        #     path = self.img_dir + name + '.tif'
        path = self.img_dir + name + '.png'
        # print(path)
        img = Image.open(path)

        return img

    def read_mask(self, name, category):
        path = self.mask_dir + name + '_instance_color_RGB.png'
        
        mask = cv2.imread(path)

        mask[mask!=category+1] = 0
        mask[mask==category+1] = 1

        return mask[:,:,0].astype(np.float32)
    '''
    def read_binary_mask(self, name, category):
        path = self.binary_mask_dir +str(category+1)+'/'+ name + '.png'
        mask = cv2.imread(path)/255

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
    
    def random_choose(self):
        class_ = np.random.choice(self.train_id_list, 1, replace=False)[0]
        cat_list = self.list_class[class_]
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)

        query_name = cat_list[sample_img_ids_1[0]]
        support_name = cat_list[sample_img_ids_1[1]]

        return support_name, query_name, class_
    
    def get_1_shot(self, idx):
        if self.count >= self.list_splite_len:
            self.random_generator.shuffle(self.list_splite)
            self.count = 0

        # query_name, class_ = self.list_splite[self.count]
        # while True:  # random sample a support data
        #     support_img_list = self.list_class[class_]
        #     support_name = support_img_list[self.random_generator.randint(0, len(support_img_list) - 1)]
        #     if support_name != query_name:
        #         break
        #
        # query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)

        
        while True:
            query_name, class_ = self.list_splite[self.count]
            while True:  # random sample a support data
                support_img_list = self.list_class[class_]
                support_name = support_img_list[self.random_generator.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)
            sum1 = query_mask.sum()
            sum2 = support_mask.sum()

            if sum1 >=  32 * 32 and sum2 >=  32 * 32:
                break
            else:
                self.count = self.count + 1
                if self.count >= self.list_splite_len:
                    self.random_generator.shuffle(self.list_splite)
                    self.count = 0

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
        #      # random sample 5 support data
        #     support_set_list = self.list_class[class_]
        #     support_choice_list = support_set_list.copy()
        #     support_choice_list.remove(query_name)
        #     support_name_list = self.random_generator.sample(support_choice_list, self.k_shot)
        #     query_img, query_mask, support_img_list, support_mask_list = self.load_frame_k_shot(support_name_list, query_name, class_)
        #     sum1 = query_mask.sum()
        #     sum2_0 = support_mask_list[0].sum()
        #     sum2_1 = support_mask_list[1].sum()
        #     sum2_2 = support_mask_list[2].sum()
        #     sum2_3 = support_mask_list[3].sum()
        #     sum2_4 = support_mask_list[4].sum()
        #     k =0
        #     if sum1 >= 32 * 32 :
        #         k=k+1
        #     if sum2_0 >= 32 * 32:
        #         k = k+1
        #     if sum2_1 >= 32 * 32:
        #         k = k+1
        #     if sum2_2 >= 32 * 32:
        #         k = k+1
        #     if sum2_3 >= 32 * 32:
        #         k = k+1
        #     if sum2_4 >= 32 * 32:
        #         k = k+1
        #
        #     if k==6:
        #         break
        #     else:
        #         self.count = self.count + 1
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
        return  self.len

    def __getitem__(self, idx):
        # support_name, query_name, class_ = self.random_choose()

        # query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)

        # if self.transform is not None:
        #     query_img, query_mask = self.transform(query_img, query_mask)
        #     support_img, support_mask = self.transform(support_img, support_mask)

        # self.count = self.count + 1
        if self.k_shot==1:
            query_img, query_mask, support_img, support_mask, class_, size,query_name  = self.get_1_shot(idx)# , size
            # label_new,class_1 = self.num_to_label(class_)
        else:
            query_img, query_mask, support_img, support_mask, class_, size,query_name = self.get_k_shot(idx)  # , size
            # label_new,class_1 = self.num_to_label(class_)
        return query_img, query_mask, support_img, support_mask,class_,size
