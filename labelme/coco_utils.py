from types import new_class
import numpy as np
import os
import tqdm
# def load_coco_txt_file(coco_file):
#     infos = []
#     if not os.path.exists(coco_file):
#         return infos
#     with open(coco_file, 'r') as f:
#         lines = f.readlines()
#         for l in lines:
#             item = l.strip('\n').split()
#             if len(item) < 6 :
#                 print('got erro:',l,coco_file)
#                 continue
#             val = [float(v) for v in item]
#             infos.append(val)
#     return infos
def load_coco_txt_file(coco_file):
    coco_annos = []
    coco_seg_pts =[]
    with open(coco_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if len(l.strip()) < 8:
                continue
            items = l.strip('\n').split(';')
            # print('items:',items,l)
            annos = [float(v) for v in items[0].split()]
            seg_pts = []
            for k in range(1,len(items)):
                seg_ptsk = [float(v) for v in items[k].split()]
                if len(seg_ptsk) >8:
                    seg_pts.append(seg_ptsk)
            
            coco_annos.append(annos)
            coco_seg_pts.append(seg_pts)
    return coco_annos

#label ctx cty w h track
def extract_coco_info(txt_file):
    infos = load_coco_txt_file(txt_file)

def write_coco_file(file_name, info_list,mode='w'):
    with open(file_name, mode) as f:
        for info in info_list:
            strinfo = '{0} '.format(int(info[0])) +' '.join(['{0:.4f}'.format(v) for v in info[1:]])
            f.write(strinfo+'\n')

def interpolate_box_info(info0,info1):
    new_infos = []
    frm_0 = info0[0]
    frm_1 = info1[0]

    if frm_1 - frm_0<2:
        return new_infos
    print('to interpolate:',frm_0,frm_1)
    coco_info0 = info0[1]
    box0 = np.array(info0[1][1:5])
    box1 = np.array(info1[1][1:5])
    # scale = 1.0/(frm_1-frm_0)
    v_box = (box1 - box0)/(frm_1-frm_0)
    for i in range(frm_0+1,frm_1):
        boxi = box0 + (i-frm_0)*v_box
        coco_infoi = [coco_info0[0],*boxi.tolist(),coco_info0[5]]
        new_infos.append((i,coco_infoi))
    return new_infos

def write_frame_infos(label_dir,frame_infos,mode):
    pbar = tqdm.tqdm(frame_infos.items())
    for k,v in pbar:
        dstf = label_dir + '/frm_{0:04d}.txt'.format(k)
        if len(v) == 0:
            continue
        write_coco_file(dstf,v,mode)


def process_frame_infos(label_dir,frame_ids,track_maps,dst_lb):
    frame_infos = []
    for i in frame_ids:
        lbf = label_dir + '/frm_{0:04d}.txt'.format(i)
        frame_infos.append(load_coco_txt_file(lbf))

    #generate a map for track
    track_infos = {}
    # print('frameinfos:',frame_infos)
    # print('trackmaps:',track_maps)
    only_del = False
    if track_maps != None and dst_lb == -1:
        only_del =True
    for idx,frm_info in enumerate(frame_infos):        
        has_modify = False
        frame_track = {}
        for box_info in frm_info:
            # print('idx:',idx,box_info)
            trackid = int(box_info[5])
            
            ##update track and label info
            if track_maps != None and trackid in track_maps:
                has_modify = True
                if dst_lb == -1:
                    continue

                box_info[5] = track_maps[trackid]
                box_info[0] = dst_lb
                trackid = track_maps[trackid]
                # print('modify:',box_info)
                
            elif track_maps != None and trackid in track_maps.values() and int(box_info[0])!=dst_lb:
                assert(dst_lb != -1)
                box_info[0] = dst_lb
                has_modify = True

            if trackid not in frame_track:
                frame_track[trackid] = [(frame_ids[idx],box_info)]
            else:
                frame_track[trackid].append((frame_ids[idx],box_info))
        if only_del and has_modify==False:
            continue
        else:
            for k,v in frame_track.items():
                if k in track_infos:
                    track_infos[k].extend(v)
                else:
                    track_infos[k] = v
    ##to process each track
    newinfos = []
    for k,v in track_infos.items():
        if only_del:
            continue
        if track_maps != None and k not in track_maps.values():
            continue
        
        num_frms = len(v)
        #interpolate box        
        for i in range(num_frms-1):            
            newinfos.extend(interpolate_box_info(v[i],v[i+1]))


    #generate frame based infos
    #if only interpolate mode,original data would not be modified
    if track_maps != None:
        frame_id_infos = {} 
        for k,v in track_infos.items():
            for info in v:
                frm = info[0]
                if frm not in frame_id_infos:
                    frame_id_infos[frm] = [info[1]]
                else:
                    frame_id_infos[frm].append(info[1])
        
        write_frame_infos(label_dir,frame_id_infos,'w')

    frame_id_infos_append = {} 
    for info in newinfos:
        frm = info[0]
        if frm not in frame_id_infos_append:
            frame_id_infos_append[frm] = [info[1]]
        else:
            frame_id_infos_append[frm].append(info[1])
    write_frame_infos(label_dir,frame_id_infos_append,'a')


# if __name__ == "__main__":
#     srcf = '/home/fuquan/workspace/videos/crowd_road_annos/frm_0001.txt'
#     xx = load_coco_txt_file(srcf)