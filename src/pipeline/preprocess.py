import os
import pandas as pd
import yaml

def file_path_to_abnormality(filepath):
    return bool(int(filepath[12]))

def  main():

    params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.yaml")))
    #params = yaml.safe_load(open("params.yaml"))

    params["dataset"]["montgomery_image_path"]

    montgomery_image_path = params["dataset"]["montgomery_image_path"]
    montgomery_leftmask_path = os.path.join(params["dataset"]["montgomery_mask_path"],'leftMask')
    montgomery_rightmask_path = os.path.join(params["dataset"]["montgomery_mask_path"],'rightMask')

    # Montgomery dataset - 138
    data_list = []
    for file in os.listdir(montgomery_image_path):
        if file.endswith('.png'):

            img_name = os.path.join(montgomery_image_path,file)
            left_name = os.path.join(montgomery_leftmask_path,file)
            right_name = os.path.join(montgomery_rightmask_path,file)

            data_list.append({
                'image': img_name,
                'seg_left': left_name,
                'seg_right': right_name,
                'source': 'montgomery',
                'tubercolosis': file_path_to_abnormality(file)
            })


    shenzen_image_path = params["dataset"]["shenzen_image_path"]
    shenzen_mask_path = params["dataset"]["shenzen_mask_path"]

    # Shenzen dataset - Segmentation folder only has 566 segmentations, even though there are 662 images!
    for file in os.listdir(shenzen_image_path):
        if file.endswith('.png'):

            name, ext = file.split('.')
            seg_file = name + '_mask.' + ext

            img_name = os.path.join(shenzen_image_path,file)

            # Corresponding segmentation name - _mask
            seg_name = os.path.join(shenzen_mask_path,seg_file)

            # Check if segmentation exists!
            if os.path.exists(seg_name):
                data_list.append({
                    'image': img_name,
                    'seg': seg_name,
                    'source': 'shenzen',
                    'tubercolosis': file_path_to_abnormality(file)
                })

    # Convet to csv and write to file. dvc can track this file for data versioning
    data_forcsv = pd.DataFrame.from_dict(data_list)
    # print("Number of cases:", data_forcsv.shape[0])
    pd.DataFrame.to_csv(data_forcsv,os.path.join(params["dataset"]["data_dir"], 'datalist.csv'), header = True,index = False)


if __name__ == "__main__":
    main()