import pickle as pkl
import os
from tqdm import tqdm
import time


with open('../../dataset/teacher_result/list_pair_map_train_test_2.pkl', 'rb') as file:
    data = pkl.load(file)

train = data['train']
test = data['test']


# os.system(f"rm -rf /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/style_test/*")
# os.system(f"rm -rf /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/content_test/*")

styles_set = set([d['style'] for d in test])

for style in styles_set:
    os.system(f"rm -rf /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/style_test/*")
    os.system(f"rm -rf /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/content_test/*")
    
    os.system(f"cp /opt/esilvestrov_toolkit_ds/98890/dataset/teacher_result/style/'{style}' /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/style_test/'{style}'")
    
    for i in range(len(test)):
        if test[i]['style'] == style:
            os.system(f"cp /opt/esilvestrov_toolkit_ds/98890/dataset/teacher_result/content/'{test[i]['content']}' /opt/esilvestrov_toolkit_ds/98890/fgist/STTR/inputs/content_test/'{test[i]['content']}'")
    
    os.system('CUDA_VISIBLE_DEVICES=0 python demo_sttr_image_ACCV.py')