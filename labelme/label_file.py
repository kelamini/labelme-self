import base64
from codecs import strict_errors
import contextlib
from genericpath import exists, isfile
import io
import json
import os.path as osp

import PIL.Image

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
import os
import cv2
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = ".json"

    def __init__(self, filename=None, img_root=''):
        self.shapes = []
        self.img_root = img_root
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename
        

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        # 根据 exif 翻转 image
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()     # 后缀
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load_coco(self, filename):
        info_filter = []
        img_exts = ['.jpg','jpeg']
        name = os.path.split(filename)[1]
        img = None
        img_size = None
        for ext in img_exts:
            img_path = os.path.join(self.img_root,name.replace('.txt',ext))
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)  # 读取 image
                img_size = np.array([img.shape[1],img.shape[0]])    # image 的 w, h
                break
        flags = {}
        # data={"version":"3.10.1","flags":{},"shapes":[],"imagePath":img_path,"imageHeight":img_size[1],"imageWidth":img_size[0],"imageData":None,"img_size":img_size}
        # print(data)
        shapes = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for idx, l in enumerate(lines):
            info_rect_contours = l.strip('\n').split(';')
            gp_id = idx+1   # group_id
            lb = None   # label
            isverify = None     # 添加 key: is_verify
            if len(info_rect_contours[0]) >= 5:
                shape={"shape_type":"rectangle"}
                info = info_rect_contours[0].split()
                shape["label"] = str(int(float(info[0])))
                lb = shape['label']
                box = np.array([float(v) for v in info[1:5]]).reshape(-1,2)*img_size[None,:]    # xc, yc, w, h
                box[0,:] -= box[1,:]*0.5        # x1, y1
                box[1,:] =box[0,:]+ box[1,:]    # x2, y2
                
                shape["points"] = box.tolist()
                shape['flags'] = {}
                shape['other_data'] = {}
                
                shape['group_id'] = None
                shape['is_verify'] = None   # 添加 key: is_verify
                if len(info) >= 7:
                    try:
                        gp_id = int(float(info[5]))
                        isverify = int(float(info[6]))
                    except:
                        gp_id = int(info[5])
                        isverify = int(info[6])
                        # gp_id = shape['group_id']
                    if len(info) > 7:
                        shape['other_data']['other_info'] = info[7:]
                shape['group_id'] = gp_id
                shape['is_verify'] = isverify   # 添加 key: is_verify
                shapes.append(shape)
            
            if len(info_rect_contours) > 1:
                for ctidx, ct_str in enumerate(info_rect_contours[1:]):
                    if len(ct_str) > 4:
                        ct_item = ct_str.split()
                        ct_val = np.array([float(v) for v in ct_item]).reshape(-1,2)*img_size[None,:]
                        shape={"shape_type":"polygon"}
                        shape["label"] = lb                        
                        shape["points"] = ct_val.tolist()
                        shape['flags'] = {}
                        shape['other_data'] = {}
                        
                        shape['group_id'] = gp_id*100
                        shapes.append(shape)

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = img_path
        self.imageData = self.load_image_file(img_path)
        self.filename = filename
        self.otherData = {"img_size":img_size}
        print('loaded:', filename, img_size)

    def load_json(self, filename):
        info_filter = []
        img_exts = ['.jpg','jpeg']
        name = os.path.split(filename)[1]
        img = None
        img_size = None
        for ext in img_exts:
            img_path = os.path.join(self.img_root,name.replace('.json',ext))
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)  # 读取 image
                img_size = np.array([img.shape[1],img.shape[0]])    # image 的 w, h
                break
        flags = {}
        # data={"version":"3.10.1","flags":{},"shapes":[],"imagePath":img_path,"imageHeight":img_size[1],"imageWidth":img_size[0],"imageData":None,"img_size":img_size}
        # print(data)
        
        with open(filename, 'r') as f:
            data = json.load(f)
        # print("line-json: ", data)
        # print("imgpath", img_path)
        print('loaded:', filename, img_size)


        shapes = data["shapes"]
        
        for shape in shapes:
            if "is_verify" not in shape:
                shape["is_verify"] = None
            if "other_data" not in shape:
                shape["other_data"] = {}
        print("shapes", shapes)

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = img_path
        self.imageData = self.load_image_file(img_path)
        self.filename = filename
        self.otherData = {"img_size":img_size}
        print('loaded:', filename, img_size)

    def load(self, filename):
        # 处理 txt 文件
        if self.img_root != '' and '.txt' == os.path.splitext(filename)[1].lower():
            return self.load_coco(filename)
        # 处理 json 文件
        if self.img_root != '' and '.json' == os.path.splitext(filename)[1].lower():
            return self.load_json(filename)
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "is_verify",    # 添加 key: is_verify
            "points",
            "group_id",     # track_id
            "shape_type",
            "flags",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warn(
                    "Loading JSON file ({}) of unknown version".format(
                        filename
                    )
                )
            elif version.split(".")[0] != __version__.split(".")[0]:
                logger.warn(
                    "This JSON file ({}) may be incompatible with "
                    "current labelme. version in file: {}, "
                    "current version: {}".format(
                        filename, version, __version__
                    )
                )

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            
            flags = data.get("flags") or {}
            
            imagePath = data["imagePath"]
            
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    is_verify=s.get["is_verify"],   # 添加 key: is_verify
                    flags=s.get("flags", {}),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save_coco(self,filename,shapes,imageHeight,imageWidth,imageData=None):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        img_size = np.array([imageWidth,imageHeight])
        print('save:',filename,'imgsize:',img_size)
        
        group_shapes = {}   # key=group_id: value=shape
        for shape in shapes:
            gpid = shape['group_id']
            if gpid != None and shape['shape_type'] == 'polygon':
                if gpid in group_shapes:
                    group_shapes[gpid].append(shape)
                else:
                    group_shapes[gpid] = [shape]
        
        with open(filename,'w') as f:
            for shape in shapes:
                if shape['shape_type'] == 'polygon':
                    continue    # only write rect firstly
                pts = np.array(shape['points'])/img_size[None,:]
                pts = np.clip(pts,0,0.999999999)
                ct = (pts[0,:] + pts[1,:])/2    # xc, yc
                wh = np.abs(pts[1,:] - pts[0,:])    # w, h
                str_item = shape['label'] + ' {0:.4f} {1:.4f} {2:.4f} {3:.4f}'.format(ct[0],ct[1],wh[0],wh[1])
                
                # 添加 track_id 和 is_verify
                gpid = shape['group_id']
                isverify = shape["is_verify"]   # 添加 key: is_verify
                if  gpid != None and isverify != None:  # 两个都存在
                    # need to check if is contour points
                    str_item += ' {}'.format(shape['group_id'])     # 添加 track_id
                    str_item += ' {}'.format(shape["is_verify"])    # 添加 is_verify
                
                if gpid == None and isverify != None:   # group_id 不存在 把位置空出来
                    str_item += '  '    # 添加 track_id
                    str_item += ' {}'.format(shape["is_verify"])    # 添加 is_verify
                    
                if 'other_data' in shape and 'other_info' in shape['other_data']:
                    str_item += ' '+ ' '.join(shape['other_data']['other_info'])
                # print('srcpts:',shape['points'],'\nnormpts:',pts,str_item)
                
                # poly_gpid = gpid*100
                if gpid != None and gpid*100 in group_shapes:
                    for shape in group_shapes[gpid*100]:
                        assert shape['shape_type'] == 'polygon'
                        pts = np.array(shape['points'])/img_size[None,:]
                        pts = np.clip(pts,0,0.999999999).flatten().tolist()
                        str_item += ';'+' '.join(['{0:.4f}'.format(v) for v in pts])
                    
                    
                f.write(str_item+'\n')
        self.filename = filename

    def save_json(self,flags,filename,shapes,imageHeight,imageWidth,imagePath,imageData=None):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        img_size = np.array([imageWidth,imageHeight])
        print('save:',filename,'imgsize:',img_size)

        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

        
        # group_shapes = {}   # key=group_id: value=shape
        # for shape in shapes:
        #     gpid = shape['group_id']
        #     if gpid != None and shape['shape_type'] == 'polygon':
        #         if gpid in group_shapes:
        #             group_shapes[gpid].append(shape)
        #         else:
        #             group_shapes[gpid] = [shape]
        
        # with open(filename,'w') as f:
        #     for shape in shapes:
        #         if shape['shape_type'] == 'polygon':
        #             continue    # only write rect firstly
        #         pts = np.array(shape['points'])/img_size[None,:]
        #         pts = np.clip(pts,0,0.999999999)
        #         ct = (pts[0,:] + pts[1,:])/2    # xc, yc
        #         wh = np.abs(pts[1,:] - pts[0,:])    # w, h
        #         str_item = shape['label'] + ' {0:.4f} {1:.4f} {2:.4f} {3:.4f}'.format(ct[0],ct[1],wh[0],wh[1])
                
        #         # 添加 track_id 和 is_verify
        #         gpid = shape['group_id']
        #         isverify = shape["is_verify"]   # 添加 key: is_verify
        #         if  gpid != None and isverify != None:  # 两个都存在
        #             # need to check if is contour points
        #             str_item += ' {}'.format(shape['group_id'])     # 添加 track_id
        #             str_item += ' {}'.format(shape["is_verify"])    # 添加 is_verify
                
        #         if gpid == None and isverify != None:   # group_id 不存在 把位置空出来
        #             str_item += '  '    # 添加 track_id
        #             str_item += ' {}'.format(shape["is_verify"])    # 添加 is_verify
                    
        #         if 'other_data' in shape and 'other_info' in shape['other_data']:
        #             str_item += ' '+ ' '.join(shape['other_data']['other_info'])
        #         # print('srcpts:',shape['points'],'\nnormpts:',pts,str_item)
                
        #         # poly_gpid = gpid*100
        #         if gpid != None and gpid*100 in group_shapes:
        #             for shape in group_shapes[gpid*100]:
        #                 assert shape['shape_type'] == 'polygon'
        #                 pts = np.array(shape['points'])/img_size[None,:]
        #                 pts = np.clip(pts,0,0.999999999).flatten().tolist()
        #                 str_item += ';'+' '.join(['{0:.4f}'.format(v) for v in pts])
                    
                    
        #         f.write(str_item+'\n')
        self.filename = filename

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        # print('tosave:',filename)
        # 处理 txt 文件
        if self.img_root != '' and '.txt' == os.path.splitext(filename)[1].lower():
            return self.save_coco(filename,shapes,imageHeight,imageWidth,imageData=imageData)
        # 处理 json 文件
        if self.img_root != '' and '.json' == os.path.splitext(filename)[1].lower():
            return self.save_json(flags, filename,shapes,imageHeight,imageWidth,imagePath,imageData=imageData)
        
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix or osp.splitext(filename)[1].lower() == '.txt'
