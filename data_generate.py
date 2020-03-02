# _*_ coding:utf-8 _*_
import os
import cv2
import numpy as np

#order of guarantee
class Order():
    def __init__(self,img51_path,txt_path='./order.txt'):
        self.img51_path=img51_path
        self.txt_path=txt_path

    def get_Order(self):
        if os.path.exists(self.txt_path):
            order_list=[]
            with open(self.txt_path,'r',encoding='utf-8')as file:
                content = file.read().split('\n')
                for i in content:
                    order_list.append(i)
            return order_list
        else:
            order_list=os.listdir(self.img51_path)
            order_len=len(order_list)
            with open(self.txt_path,'w+',encoding='utf-8')as file:
                for i in range(order_len):
                    if i!=order_len-1:
                        file.write(order_list[i]+'\n')
                    else:
                        file.write(order_list[i])
            return order_list

#yield data
class Generate():
    def __init__(self,img51_path='./align51_org',test51_path='./test51_org',frames_len=25,batch_size=2):
        self.batch_size=batch_size
        self.frames_len=frames_len
        self.img51_path=img51_path
        self.test51_path=test51_path
        ord=Order(self.img51_path)
        self.order_list=ord.get_Order()
        self.order_len=len(self.order_list)
        order_dict={}
        order_traverse_dict={}
        for i,j in zip(self.order_list,range(len(self.order_list))):
            order_dict[i]=j
            order_traverse_dict[j]=i
        self.order_dict=order_dict
        self.order_traverse_dict=order_traverse_dict
    def read_Sample(self,path,classes):
        img_list=[]
        for i in range(self.frames_len):
            full_path=os.path.join(path,str(i+1)+'.png')
            #print(full_path)
            curr_img=cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
            curr_img=cv2.resize(curr_img,(320,240))
            curr_array=np.array(curr_img)
            #print(curr_array.shape)
            img_list.append(curr_array)
        img_array=np.array(img_list)
        #print('img_array',img_array.shape)
        img_array=np.transpose(img_array,[1,2,0])
        assert img_array.shape[-1]==self.frames_len
        assert img_array.shape[0:2]==(240,320)
        # print(img_array.shape)
        # test=img_array[:,:,0]
        # print(test.shape)
        # cv2.imshow('teset',test)
        # cv2.waitKey(0)
        label=np.zeros(shape=[self.order_len],dtype=np.float32)
        label[classes]=1
        return img_array,label

    def get_TrainExamples(self,sample_limit=180):
        return sample_limit*self.order_len

    def get_TestExamples(self):
        return 11*self.order_len

    def get_Classes(self):
        return self.order_len

    def train_Generate(self,sample_limit=180):
        while 1:
            img_array=[]
            label_array=[]
            #random sample
            classes_random=np.random.permutation(self.order_len)
            slice_classes=classes_random[:self.batch_size]
            sample_random=np.random.permutation(sample_limit)
            slice_sample=sample_random[:self.batch_size]
            for i,j in zip(slice_classes,slice_sample):
                class_name=self.order_traverse_dict[i]
                class_path=os.path.join(self.img51_path,class_name)
                sample_path=os.path.join(class_path,os.listdir(class_path)[j])
                sample_img,sample_label=self.read_Sample(sample_path,i)
                img_array.append(sample_img)
                label_array.append(sample_label)
            img_array=np.array(img_array)
            label_array=np.array(label_array)
            #print(img_array.shape,label_array.shape)
            yield img_array,label_array

    #读取数据至内存。。扩充数据后不适用
    def train_Generate_2(self,sample_limit=90):
        img_array=[]
        label_array=[]
        for i in range(self.order_len):
            curr_path = os.path.join(self.img51_path, self.order_list[i])
            curr_list = os.listdir(curr_path)[:sample_limit]
            for j in curr_list:
                sample_path=os.path.join(curr_path,j)
                sample_img,sample_label=self.read_Sample(sample_path,i)
                #print(sample_img.shape,sample_label.shape)
                img_array.append(sample_img)
                label_array.append(sample_label)
        img_array = np.array(img_array)
        label_array = np.array(label_array)
        while 1:
            random_index=np.random.permutation(label_array.shape[0])
            for i in range(label_array.shape[0]//self.batch_size):
                batch_x,batch_y=img_array[random_index[i*self.batch_size:(i+1)*self.batch_size]],\
                                label_array[random_index[i*self.batch_size:(i+1)*self.batch_size]]
                #print(batch_x.shape,batch_y.shape)
                yield batch_x,batch_y
    #561 samples
    def test_Generate(self):
        img_array=[]
        label_array=[]
        for i in range(self.order_len):
            curr_path=os.path.join(self.test51_path,self.order_list[i])
            curr_list=os.listdir(curr_path)
            for j in curr_list:
                sample_path=os.path.join(curr_path,j)
                sample_img,sample_label=self.read_Sample(sample_path,i)
                #print(sample_img.shape,sample_label.shape)
                img_array.append(sample_img)
                label_array.append(sample_label)
        img_array=np.array(img_array)
        label_array=np.array(label_array)
        while 1:
            random_index=np.random.permutation(label_array.shape[0])
            for i in range(label_array.shape[0]//self.batch_size):
                batch_x,batch_y=img_array[random_index[i*self.batch_size:(i+1)*self.batch_size]],\
                                label_array[random_index[i*self.batch_size:(i+1)*self.batch_size]]
                #print(batch_x.shape,batch_y.shape)
                yield batch_x,batch_y



if __name__=='__main__':
    ge=Generate()
    #ge.read_Sample('align51_org/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0',0)
    ge.train_Generate()
    #ge.test_Generate()
    pass