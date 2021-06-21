from argparse import ArgumentParser
import base64
import datetime
import hashlib
import os
import cv2
from PIL import Image
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
from torch import nn
import time
import torch
import torchvision.transforms as transforms
from model import Classifier ,EfNetModel


app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'mickey23405383@gmail.com'          #
SALT = 'ntustntustntust'                        #
#########################################

# verification mickey23405383@gmail.com 437292 http://140.118.47.155:8080
# get_status mickey23405383@gmail.com 437292


def get_label_map():
    label2word = {}
    map = pd.read_csv('./label_map.csv')
    for idx,word in enumerate(map['Word']):
        label2word[idx] = word
    return label2word


def load_model():
    # 柏翰 原始資料 training
    # model1 = Classifier().to(device)
    # model1_path = './models/CNN_SGD_3e2__drop05085_300epoch.ckpt'
    # model1.load_state_dict(torch.load(model1_path))
    # model1.eval()

    # model2 = Classifier().to(device)
    # model2_path = './models/CNN_SGD_3e2__drop05085_500epoch.ckpt'
    # model2.load_state_dict(torch.load(model2_path))
    # model2.eval()

    # pretrained on .945 axot
    # model2 = EfNetModel(num_classes=801,pretrained_path='./models/eff_ori128_t_d9.pth').to(device)
    # model2.eval()
    # # axot 0.9455
    # model3 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_f94550.pth').to(device)
    # model3.eval()

    # pretrained on .945 axot using 615 data
    model4 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_0615.pth').to(device)
    model4.eval()


    # # 柏翰用615 data 
    # model5 = Classifier().to(device)
    # model5_path = './models/CNN_SGD_3e3_drop05085_300epoch_2_day1.ckpt'
    # model5.load_state_dict(torch.load(model5_path))
    # model5.eval()
    # # 柏翰用615 data
    # model6 = Classifier().to(device)
    # model6_path = './models/CNN_SGD_3e3_drop05085_300epoch_day1.ckpt'
    # model6.load_state_dict(torch.load(model6_path))
    # model6.eval()

    # # #柏翰的 old + 615 as training data
    # model7 = Classifier().to(device)
    # model7_path = './models/CNN_SGD_3e2_drop05085_300epoch_mix_data.ckpt'
    # model7.load_state_dict(torch.load(model7_path))
    # model7.eval()

    # #
    # model8 = Classifier().to(device)
    # model8_path = './models/CNN_SGD_3e3_drop05085_300epoch_mix_data_day2.ckpt'
    # model8.load_state_dict(torch.load(model8_path))
    # model8.eval()

    # 0615 + 0616 + olddata
    model9 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_0616_with_old_data.pth').to(device)
    model9.eval()


    # 
    model10 = Classifier().to(device)
    model10_path = './models/CNN_SGD_3e2_drop05085_300epoch_2_weight_day1.ckpt'
    model10.load_state_dict(torch.load(model10_path))
    model10.eval()

    model11 = Classifier().to(device)
    model11_path = './models/CNN_SGD_3e3_drop05085_300epoch_2_weight_day1.ckpt'
    model11.load_state_dict(torch.load(model11_path))
    model11.eval()

    model12 = Classifier().to(device)
    model12_path = './models/CNN_SGD_3e2_drop05085_200epoch_all_data_weight.ckpt'
    model12.load_state_dict(torch.load(model12_path))
    model12.eval()

    # 615 616 only
    model13 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_615_616.pth').to(device)
    model13.eval()

    models = [
        ('eff', model4),
        ('eff', model9), 
        ('eff', model13),
        ('mouth', model10),
        ('mouth', model11),
        ('mouth', model12)
    ]


    return models



def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######    
    PIL_img = Image.fromarray(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
    tfms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    img = tfms(PIL_img).unsqueeze(0).to(device)
    with torch.no_grad():
        first = True
        soft = nn.Softmax()
        votes = {}
        for model_type , model in models:
            output = model(img)
            logits = soft(output)
            if model_type == 'eff':
                if 'eff' in votes:
                    votes['eff'] += logits
                else:
                    votes['eff'] = logits

            elif model_type == 'mouth':
                if 'mouth' in votes:
                    votes['mouth'] += logits
                else:
                    votes['mouth'] = logits

        # eff vote
        logits_eff = votes['eff'] / num_of_eff_models
        max_prob_eff , max_idx_eff = torch.max(logits_eff.cpu(),dim=-1)
        should_be_isnull_eff = max_prob_eff < eff_threshold

        # mouth vote
        logits_mouth = votes['mouth'] / num_of_mouth_models
        max_prob_mouth , max_idx_mouth = torch.max(logits_mouth.cpu(),dim=-1)
        should_be_isnull_mouth = max_prob_mouth < mouth_threshold

        should_be_isnull = torch.logical_and(should_be_isnull_mouth,should_be_isnull_eff)

        
        
        if should_be_isnull.item():
            prediction = 'isnull'
        else:
            logits =  logits_mouth + logits_eff
            pred = torch.argmax(logits,dim=-1)
            prediction  =  label_map[pred.item()]



    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def predict_detail(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######    
    PIL_img = Image.fromarray(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
    tfms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    img = tfms(PIL_img).unsqueeze(0)
    with torch.no_grad():
        probs = []
        sum_prob = None
        soft = nn.Softmax()
        for i, (model_type,model) in enumerate(models):
            prob = soft(model(img.to(device)))
            probs.append(prob)

        preds_word = [label_map[torch.argmax(prob, dim=-1).item()] for prob in probs]
        probs = [torch.max(prob, dim=1)[0].item() for prob in probs]


    return preds_word, probs





def logging(image_64_encoded, label=None,test=False):
    date = datetime.datetime.today().strftime('%m_%d')
    if test:
        dir_path = os.path.join(os.getcwd(),'logs_test')
        dir_path = os.path.join(dir_path,date)
    else:
        dir_path = os.path.join(os.getcwd(),'logs')
        dir_path = os.path.join(dir_path,date)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_id = str(len(os.listdir(dir_path)))
    if label == 'isnull':
        label = 'N'

    imgdata = base64.b64decode(image_64_encoded)
    if label:
        filename = os.path.join(dir_path, data_id + '_' + label + '.jpg')  
    else:
        filename = os.path.join(dir_path, data_id + '.jpg')  
    with open(filename, 'wb') as f:
        f.write(imgdata)
    


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    # server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    server_timestamp = int(time.time())
    logging(image_64_encoded, answer)

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


@app.route('/inference_testing', methods=['POST'])
def inference_testing():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    start_time = time.time()
    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)
    
    try:
        answer = predict(image)
        preds_word, probs = predict_detail(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    
    end_time = time.time()

    logging(image_64_encoded, answer , test=True)
    return jsonify({
        'answer': answer,
        'inference_time' : end_time - start_time,
        'detail': {
            f"model[{i}]":{
                'pred_word': w,
                'prob': p
            } for i, (w, p) in enumerate(zip(preds_word, probs)) 
        }
    })


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8787, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    models = load_model()
    num_of_models = len(models)
    num_of_eff_models = 3
    num_of_mouth_models = 3
    eff_threshold = 0.9
    mouth_threshold = 0.8
    
    label_map = get_label_map()

    app.run(host='0.0.0.0',debug=options.debug, port=options.port)
