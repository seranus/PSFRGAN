from options.test_options import TestOptions
import os
import cv2
import numpy as np 
from tqdm import tqdm
from skimage import transform as trans
from skimage import io
from models import create_model
import torch
from utils import utils
from pathlib import Path
from PIL import Image
import json
from pathlib import Path

def enhance_faces(LQ_faces, model):
    hq_faces = []
    lq_parse_maps = []
    for lq_face in LQ_faces:
        with torch.no_grad():
            lq_tensor = torch.tensor(lq_face.transpose(2, 0, 1)) / 255. * 2 - 1
            lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)
            parse_map, _ = model.netP(lq_tensor)
            parse_map_onehot = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            _, output_SR = model.netG(lq_tensor, parse_map_onehot)
        hq_faces.append(utils.tensor_to_img(output_SR))
        lq_parse_maps.append(utils.color_parse_map(parse_map_onehot)[0])
    return hq_faces, lq_parse_maps

def def_models(opt):
    model = create_model(opt)
    model.load_pretrain_models()
    model.netP.to(opt.device)
    model.netG.to(opt.device)
    return model

def landmark_68_to_5(landmarks):
    lan_5 = np.array([landmarks[45], landmarks[42], landmarks[36], landmarks[39], landmarks[34]])
    return lan_5

if __name__ == '__main__':
    opt = TestOptions().parse()

    base_path = Path(__file__).parent

    opt.parse_net_weight = os.path.join(base_path, 'pretrain_models', 'parse_multi_iter_90000.pth')
    opt.psfr_net_weight = os.path.join(base_path, 'pretrain_models', 'psfrgan_latest_net_G.pth')

    input_path = opt.src_dir
    output_path = opt.results_dir

    enhance_model = def_models(opt)
    enhance_model = def_models(opt)

    reference = np.load(os.path.join(Path(__file__).parent, 'FFHQ_template.npy')) / 2
    out_size = (512, 512) 

    img_names = os.listdir(input_path)
    img_names.sort()

    for i, img_name in enumerate(tqdm(img_names)):
        # read image
        data_input = input('%REQFILE%$' + img_name)
        if len(data_input) < 2:
            continue

        try:
            # align image
            # set numpy landmarks
            landmarks_string = json.loads(data_input)
            landmarks = np.array(landmarks_string)

            A_paths = os.path.join(input_path, img_name)
            img = Image.open(A_paths).convert('RGB')
            img_width, img_height = img.size

            # crop
            source = landmark_68_to_5(landmarks)
            tform = trans.SimilarityTransform()                                                                                                                                                  
            tform.estimate(source, reference)
            M = tform.params[0:2,:]
            array_img = np.array(img)
            crop_img = cv2.warpAffine(array_img, M, out_size)

            # enhance
            hq_faces, lq_parse_maps = enhance_faces([crop_img], enhance_model)

            image_numpy = cv2.warpAffine(hq_faces[0], M, img.size, array_img, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )

            image_pil = Image.fromarray(image_numpy)
            image_pil.save(os.path.join(output_path, img_name))
        except Exception as e:
            print(r'%ERROR%$Error in enhancing this image: {}'.format(str(e)))

            continue