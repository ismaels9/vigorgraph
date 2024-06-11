from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os
import json
import shutil
import logging
import atexit
import cv2 as cv
import glob
import argparse
from findunion import concatenate_lines
from utils import draw_line
from typing import Tuple, List
from algorithms import edge_linking, rdp
from preprocessing import find_background_color_range, find_background_mask
from skimage.util import img_as_ubyte
from skimage import morphology
from seedlings import SeedlingSolver
import numpy as np
from yolo import YOLOProxy
from functools import reduce
from gevent.pywsgi import WSGIServer
from gevent import monkey
import uuid  # Importação para gerar UUIDs

atexit.register(cv.destroyAllWindows)
monkey.patch_all()

app = Flask(__name__)
CORS(app)  # Isso permite solicitações de todas as origens
UPLOAD_FOLDER_BASE = os.path.expanduser('~/Documentos/vigorgraph')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER_BASE):
    os.makedirs(UPLOAD_FOLDER_BASE)

SHOW_IMAGE = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_lines(img, pts, color, pt_color):
    draw_line(img, pts, color)
    for i in range(len(pts)):
        cv.circle(img, pts[i], 1, pt_color, 1)

def find_lines(raiz_prim: np.ndarray, hipocotilo: np.ndarray, epsilon=20) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    raiz_prim = img_as_ubyte(morphology.skeletonize(raiz_prim))
    hipocotilo = img_as_ubyte(morphology.skeletonize(hipocotilo))
    raiz_prim_links = edge_linking(raiz_prim)
    hipocotilo_links = edge_linking(hipocotilo)
    raiz_prim_links_rdp = [rdp(link, epsilon=epsilon) for link in raiz_prim_links]
    hipocotilo_links_rdp = [rdp(link, epsilon=epsilon) for link in hipocotilo_links]
    raiz_prim_links_rdp = concatenate_lines(raiz_prim_links_rdp, threshold=20)
    hipocotilo_links_rdp = concatenate_lines(hipocotilo_links_rdp, threshold=20)
    return raiz_prim_links_rdp, hipocotilo_links_rdp

def resize_image(image, max_size):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv.resize(image, new_size, interpolation=cv.INTER_AREA)
    return image

def rm_bg(img: np.ndarray):
    color_range = find_background_color_range(img)
    input_img_wo_background = find_background_mask(img, color_range)
    return input_img_wo_background

def find_seed_blobs(input_img_wo_background: np.ndarray, iterations=5):
    input_img_wo_background = cv.erode(input_img_wo_background, np.ones((3, 3)), iterations=iterations)
    return [cnt for cnt in cv.findContours(input_img_wo_background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0] if cv.contourArea(cnt) < 1000]

def run(input_img_paths):
    model = YOLOProxy('./models/yolov8/best.pt')
    imgs = []
    orig_img_sizes = []
    for img_path in input_img_paths:
        img = cv.imread(img_path)
        imgs.append(resize_image(img, 1200))

    logging.getLogger().setLevel(logging.WARNING)

    raiz_prim_masks, hipocotilo_masks = model.predict(imgs, imgsz=1216)
    output = dict()
    for hipocotilo_mask, raiz_prim_mask, img, im_path in zip(raiz_prim_masks, hipocotilo_masks, imgs, input_img_paths):
        raiz_prim_links, hipocotilo_links = find_lines(raiz_prim_mask, hipocotilo_mask, epsilon=1)
        input_img_wo_background = rm_bg(img) 
        contours  = find_seed_blobs(input_img_wo_background, iterations=8)
        cotyledone = []
        for ct in contours:
            M = cv.moments(ct)
            cX = int(M["m10"] / (M["m00"] + 0.001))
            cY = int(M["m01"] / (M["m00"] + 0.001))
            cotyledone.append((cX,cY))

        ss = SeedlingSolver(raiz_prim_links, hipocotilo_links, np.array(cotyledone), max_cost=100)
        seedlings = ss.match()

        # turning from y,x shape to x,y as coordinates are xy in seedlings.
        conversion_ratio = (1/np.array(img.shape[:2]))[::-1]
        
        info = {
            'links': {
                i: {
                    'hipocotilo': (sdl.hipocotilo * conversion_ratio).tolist() if sdl.hipocotilo is not None else None,
                    'raiz_prim': (sdl.raiz_prim * conversion_ratio).tolist() if sdl.raiz_prim is not None else None
                }
                for i,sdl in enumerate(seedlings)
            },
            'numero_plantulas': len(seedlings),
            'numero_plantuas_ngerm': reduce(lambda acc, sdl: int(sdl.is_dead()) + acc, seedlings, 0) 
        }
        output[im_path] = info
        if SHOW_IMAGE:
            seedlings_drawed = np.zeros_like(img)
            for sdl in seedlings:
                seedlings_drawed = sdl.draw(seedlings_drawed)

            overlayed_img = cv.addWeighted(img, 0.5,seedlings_drawed, 0.5, 0)
            while True:
                cv.imshow('overlayed_img', overlayed_img)
                cv.imshow('hip_mask', hipocotilo_mask)
                cv.imshow('raiz_mask', raiz_prim_mask)
                key = cv.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

    """
    overlayed_img = cv.addWeighted(input_img, 0.5,seedlings_drawed, 0.5, 0)

    if SHOW_IMAGE:
        while True:
            cv.imshow('seedlings_drawed', seedlings_drawed)
            cv.imshow('overlay', overlayed_img)
            cv.imshow('blob', blobs_image)
            cv.imshow('blobs', blobs_image)
            cv.imshow('gd_img', gd_img)
            #cv.imshow('raiz_prim_ske', raiz_prim_skeleton)
            #cv.imshow('hip_ske', hip_skeleton)
            cv.imshow('linked_raiz_prim', linked_raiz_prim)
            cv.imshow('linked_hipocotilo', linked_hipocotilo)
            #cv.imshow('seed blobs', input_img_wo_background)

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
    """
    return output

# Rota para receber uma imagem via POST
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    files = request.files.getlist('images')
    folder_uuid = str(uuid.uuid4())
    upload_folder = os.path.join(UPLOAD_FOLDER_BASE, folder_uuid)
    os.makedirs(upload_folder)
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = os.path.join(upload_folder, file.filename)
            file.save(filename)
            filenames.append(filename)
        else:
            shutil.rmtree(upload_folder)  # Remove o diretório se um arquivo não for permitido
            return jsonify({'error': 'File type not allowed'}), 400
    return jsonify({'message': 'Images uploaded successfully', 'filenames': filenames, 'folder': folder_uuid}), 200

# Novo endpoint para processar imagens usando a lógica fornecida
@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        folder_uuid = request.json.get('folder')
        if not folder_uuid:
            return jsonify({'error': 'Folder not specified'}), 400
        upload_folder = os.path.join(UPLOAD_FOLDER_BASE, folder_uuid)
        if not os.path.exists(upload_folder):
            return jsonify({'error': 'Folder does not exist'}), 400
        image_paths = [os.path.join(upload_folder, filename) for filename in os.listdir(upload_folder) if allowed_file(filename)]
        if not image_paths:
            return jsonify({'error': 'No images found in the specified folder'}), 400
        result = run(image_paths)
        shutil.rmtree(upload_folder)  # Remove o diretório após o processamento
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Comentado para uso com Gunicorn
# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
