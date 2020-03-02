# _*_ coding:utf-8 _*_
import cv2
import os
import shutil
import math

from PIL import Image

class TransVideo():
    def __init__(self,src_dir='./hmdb51_org',target_dir='./extend/img51_org',
                 align_dir='./extend/align51_org',test_sample_dir='./extend/test51_org',
                 true_test_dir='./test51_org',frames_len=25):
        self.src_dir=src_dir
        self.test_sample_dir=test_sample_dir
        self.target_dir=target_dir
        self.true_test_dir=true_test_dir
        if os.path.exists(self.target_dir):
            pass
        else:
            os.mkdir(self.target_dir)
        #self.interval=interval
        self.align_dir=align_dir
        if os.path.exists(self.align_dir):
            pass
        else:
            os.mkdir(self.align_dir)
        self.frames_len=frames_len
        self.multi_Video()
        self.update_all_Imgs()
        self.test_Division()
        self.image_Extension()
        self.copy_File()

    def get_Interval(self,true_frames):
        rate=round(true_frames/self.frames_len)     #round
        if rate>=1:
            return rate
        else:
            return int(1)

    def single_Trans(self,video_name,target_path):
        #get video len
        cap_count=cv2.VideoCapture(video_name)
        frames_len=0
        while(cap_count.isOpened()):
            _,frame=cap_count.read()
            if frame is None:
                break
            frames_len+=1
        interval=self.get_Interval(frames_len)
        #read video
        cap = cv2.VideoCapture(video_name)
        frame_count = 1
        img_count = 1
        while(cap.isOpened()):
            #read frames
            ret,frame=cap.read()
            if frame is None:
                cap.release()
                cv2.destroyAllWindows()
                break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('frame',gray)
            # if cv2.waitKey(1)&0xFF==ord('q'):
            #     break
            if frame_count%interval==0:
                curr_path=os.path.join(target_path,str(img_count)+'.png')
                cv2.imwrite(curr_path,gray)
                img_count+=1
            frame_count+=1

    def multi_Video(self):
        target51=os.listdir(self.target_dir)
        if len(target51)==51:
            print('save successful')
            return
        hmdb51=os.listdir(self.src_dir)
        hmdb51_len=len(hmdb51)
        for i in range(hmdb51_len):
            #make target children dir
            child_target=os.path.join(self.target_dir,hmdb51[i])
            os.mkdir(child_target)
            #get current grandson_lists
            child_current=os.path.join(self.src_dir,hmdb51[i])
            grandson_list=os.listdir(child_current)
            grandson_len=len(grandson_list)
            for j in range(grandson_len):
                #make target grandson_lists
                grandson_target=os.path.join(child_target,grandson_list[j])[:-4] #+'extend'
                os.mkdir(grandson_target)
                #get_all_images
                grandson_current=os.path.join(child_current,grandson_list[j])
                self.single_Trans(grandson_current,grandson_target)

    #k=imgs_len/self.frames_len+0.001 value_original=math.round(i/k)
    def update_single_Imgs(self,imgs_path,target_path):
        imgs_list=os.listdir(imgs_path)
        imgs_len=len(imgs_list)
        k=imgs_len/self.frames_len  #50/25  k=2
        count=1
        if k>=1:
            for i in range(imgs_len):
                if i==0:
                    value_original=round(i/k)    #round
                    value_previous=value_original
                    select_img=cv2.imread(os.path.join(imgs_path,str(i+1)+'.png'))
                    cv2.imwrite(os.path.join(target_path,str(count)+'.png'),select_img)
                    count+=1
                else:
                    value_original=round(i/k)   #round
                    if value_original==value_previous:
                        pass
                    else:
                        value_previous=value_original
                        select_img = cv2.imread(os.path.join(imgs_path, str(i+1) + '.png'))
                        cv2.imwrite(os.path.join(target_path, str(count) + '.png'), select_img)
                        count+=1
                        if count>self.frames_len:
                            break
        else:
            for i in range(self.frames_len):
                select_value=round(i*k+1)   #round
                select_img=cv2.imread(os.path.join(imgs_path,str(select_value)+'.png'))
                cv2.imwrite(os.path.join(target_path,str(count)+'.png'),select_img)
                count+=1

    def update_all_Imgs(self):
        align51=os.listdir(self.align_dir)
        if len(align51)==51:
            print('align successful')
            return
        img51=os.listdir(self.target_dir)
        img51_len=len(img51)
        for i in range(img51_len):
            #make target child dir
            child_target=os.path.join(self.align_dir,img51[i])
            os.mkdir(child_target)
            #get current grandsons_lists
            child_current=os.path.join(self.target_dir,img51[i])
            grandson_list=os.listdir(child_current)
            grandson_len=len(grandson_list)
            for j in range(grandson_len):
                #make target grandson list
                grandson_target=os.path.join(child_target,grandson_list[j])
                os.mkdir(grandson_target)
                #align all images
                grandson_current=os.path.join(child_current,grandson_list[j])
                self.update_single_Imgs(grandson_current,grandson_target)
                #judge error happened
                assert len(os.listdir(grandson_target))==self.frames_len

    #数据水平翻转扩充
    def image_Flip(self,source_path,target_path):
        source_list=os.listdir(source_path)
        source_len=len(source_list)
        for i in range(source_len):
            image=Image.open(os.path.join(source_path,str(i+1)+'.png'))
            # 图像水平翻转
            image_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_trans.save(os.path.join(target_path,str(i+1)+'.png'))

    # 数据水平翻转扩充
    def image_Extension(self):
        align_51=os.listdir(self.align_dir)
        if len(os.listdir(os.path.join(self.align_dir,align_51[0])))>180:
            print('flip_extension successful')
            return
        align_51_len=len(align_51)
        for i in range(align_51_len):
            align_51_path=os.path.join(self.align_dir,align_51[i])
            align_sample_list=os.listdir(align_51_path)
            align_sample_len=len(align_sample_list)
            for j in range(align_sample_len):
                curr_path=os.path.join(align_51_path,align_sample_list[j])
                flip51_path=curr_path+'_Flip'
                if os.path.exists(flip51_path):
                    pass
                else:
                    os.mkdir(flip51_path)
                self.image_Flip(curr_path,flip51_path)

    #测试集样本分离
    def sample_Division(self,source_dir,target_dir):
        shutil.copytree(source_dir,target_dir)
        shutil.rmtree(source_dir)

    #测试集样本分离
    def test_Division(self,sample_limit=90):
        align_51=os.listdir(self.align_dir)
        if os.path.exists(self.test_sample_dir):
            print('test_division successful')
            return
        else:
            os.mkdir(self.test_sample_dir)
        align_51_len=len(align_51)
        for i in range(align_51_len):
            align_51_path=os.path.join(self.align_dir,align_51[i])
            test_sample_path=os.path.join(self.test_sample_dir,align_51[i])
            os.mkdir(test_sample_path)
            align_sample_list=os.listdir(align_51_path)[sample_limit:sample_limit+11]
            ############
            true_test_path=os.path.join(self.true_test_dir,align_51[i])
            align_sample_list=os.listdir(true_test_path)
            ############
            align_sample_len=len(align_sample_list)
            assert align_sample_len==11
            for j in range(align_sample_len):
                align_sample_path=os.path.join(align_51_path,align_sample_list[j])

                test_11_each_path=os.path.join(test_sample_path,align_sample_list[j])
                #os.mkdir(test_11_each_path)
                self.sample_Division(align_sample_path,test_11_each_path)

    def copy_File(self,source_dir='./extend/align51_org',target_dir='./align51_org'):
        extend_list=os.listdir(source_dir)
        extend_len=len(extend_list)
        if extend_len==0:
            print('move successful')
            return
        for i in range(extend_len):
            extend_path=os.path.join(source_dir,extend_list[i])
            sample_list=os.listdir(extend_path)
            target_path=os.path.join(target_dir,extend_list[i])
            for j in range(len(sample_list)):
                shutil.copytree(os.path.join(extend_path,sample_list[j]),
                                os.path.join(target_path,sample_list[j]))
                # print(os.path.join(extend_path,sample_list[j]))
                # print(os.path.join(target_path,sample_list[j]))

            shutil.rmtree(extend_path)


if __name__=='__main__':
    def trans_name(path='./extend/align51_org'):
        extend51_list=os.listdir(path)
        extend51_list_len=len(extend51_list)
        for i in range(extend51_list_len):
            classes_path=os.path.join(path,extend51_list[i])
            sample_list=os.listdir(classes_path)
            for j in range(len(sample_list)):
                sample_path=os.path.join(classes_path,sample_list[j])
                os.rename(sample_path,sample_path+'_extend')
    #trans_name()
    tv=TransVideo()

