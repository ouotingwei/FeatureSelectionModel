import numpy as np
import os
import json
# 指定根目錄路徑

def load_npy_file(file_path):
    """
    讀取 .npy 文件並返回數據。如果文件不存在或讀取失敗，則打印錯誤信息。
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    try:
        data = np.load(file_path)
        print(f"Data loaded successfully from '{file_path}'!")
        return data
    except Exception as e:
        print(f"An error occurred while loading the file '{file_path}': {e}")
        return None

def traverse_and_load_npy(root_directory,i):
    """
    遍歷根目錄中的所有子目錄，讀取每個子目錄中的 'error.npy'、'input_uv.npy' 和 'input_XYZ.npy' 文件。
    """
    all_error_data = []
    all_XYZ_data = []
    all_3d_data = []
    all_response_data = []
    all_size_data = []

    
    error_file_path = os.path.join(root_directory, str(i), 'error.npy')
    input_XYZ_file_path = os.path.join(root_directory, str(i), 'input_XYZ.npy')
    input_3d_file_path = os.path.join(root_directory, str(i), 'input_diversity3d.npy')
    input_response_file_path = os.path.join(root_directory, str(i), 'input_response.npy')
    input_size_file_path = os.path.join(root_directory, str(i), 'input_size.npy')
    print(error_file_path)

    error_data = load_npy_file(error_file_path)
    XYZ_data = load_npy_file(input_XYZ_file_path)
    d3d_data = load_npy_file(input_3d_file_path)
    res_data = load_npy_file(input_response_file_path)
    size_data = load_npy_file(input_size_file_path)

    if error_data is not None:
        all_error_data.append(error_data)
    if XYZ_data is not None:
        all_XYZ_data.append(XYZ_data)
    if d3d_data is not None:
        all_3d_data.append(d3d_data)
    if res_data is not None:
        all_response_data.append(res_data)
    if size_data is not None:
        all_size_data.append(size_data)

    if all_error_data:
        all_error_data = np.concatenate(all_error_data)
    else:
        all_error_data = np.array([])

    if all_XYZ_data:
        all_XYZ_data = np.concatenate(all_XYZ_data)
    else:
        all_XYZ_data = np.array([])
    
    if all_3d_data:
        all_3d_data = np.concatenate(all_3d_data)
    else:
        all_3d_data = np.array([])
    
    if all_response_data:
        all_response_data = np.concatenate(all_response_data)
    else:
        all_response_data = np.array([])

    if all_size_data:
        all_size_data = np.concatenate(all_size_data)
    else:
        all_size_data = np.array([])

    #print("Total error data length:", len(all_error_data))
    #print("Total uv data length:", len(all_uv_data))
    #print("Total XYZ data length:", len(all_XYZ_data))
    return all_XYZ_data,all_3d_data, all_response_data, all_size_data,all_error_data

def squeeze_clip(x, min_x, max_x, max_num=1):

    x_scaled = (x - min_x) / (max_x - min_x) * max_num

    return x_scaled

if __name__ == "__main__":
    # uv,xyz,error=traverse_and_load_npy()
    max_x=-np.inf
    max_y=-np.inf
    max_z=-np.inf
    min_x=np.inf
    min_y=np.inf
    min_z=np.inf

    data=[]
    for j in range(1,677):
        text=[]
        label=[]
        xyz,d3d, res, siz,err=traverse_and_load_npy('/home/wei/deep_feature_selection/training_data/loris_800',j)
        max_xyz=np.max(xyz,axis=0)
        min_xyz=np.min(xyz,axis=0)

        if(max_xyz[0]>max_x):
            max_x=max_xyz[0]
        if(max_xyz[1]>max_y):
            max_y=max_xyz[1]
        if(max_xyz[2]>max_z):
            max_z=max_xyz[2]

        if(min_xyz[0]<min_x):
            min_x=min_xyz[0]
        if(min_xyz[1]<min_y):
            min_y=min_xyz[1]
        if(min_xyz[2]<min_z):
            min_z=min_xyz[2]
            
        # print(max_uv,"\n",min_uv,"\n",max_xyz,"\n",min_xyz)
        for i in range(len(xyz)):
            # text.append(round(uv[i][0]+5*uv[i][1]+25*xyz[i][0]+125*xyz[i][1]+625*xyz[i][2]))
            text.append([
                        #  squeeze_clip(xyz[i][0],-2.525,3.638),
                        #  squeeze_clip(xyz[i][1],-2.067,1.624),
                         squeeze_clip(d3d[i][0],0,1),
                         squeeze_clip(xyz[i][2],0.376,4.328),
                         squeeze_clip(res[i][0],0,150),
                         siz[i][0]])

            if err[i]<1:
                label.append(1)
            else:
                label.append(0)

        # for i in range(len(text),1024):
        #     text.append([0,0,0,0])
        #     label.append(0)
        data.append({
            "tokens":text,
            "ner_tags":label
        })

    with open('custom_ner_dataset.json', 'w') as f:
        json.dump(data, f)
    #print(max_u,",",max_v,",",min_u,",",min_v,",",max_x,",",max_y,",",max_z,",",min_x,",",min_y,",",min_z,",")
    # 打印检查生成的 JSON 数据
    # print(json.dumps(data))