import os
import os.path as osp
import glob
import numpy as np
import argparse
import pandas as pd

def build_list(args):
    data_dir = args.tiny_imgnet_dir
    #Read all the available classes
    cls_name_dict = {}
    with open(osp.join(data_dir, "wnids.txt")) as fi:
        lines = fi.readlines()
        lines = [i.strip() for i in lines]
        for i in range(len(lines)):
            cls_name_dict[i] = lines[i]
    
    n_total_cls = len(cls_name_dict)
    
    def get_all_images(class_id, subset = "train"):
        class_name = cls_name_dict[class_id]
        imgs_lst   = sorted(glob.glob(osp.join(data_dir, subset, class_name,  "images", "*.JPEG")))
        imgs_lst   = [i.replace(data_dir,"") for i in imgs_lst]
        return imgs_lst

    def build_img_lists(subset = "train"):
        all_imgs_path = []
        all_imgs_cls  = []
        all_cls_imgs  = []
        for cls_id in range(n_total_cls):
            cls_imgs_path = get_all_images(cls_id, subset)
            next_idx      = len(all_imgs_path)
            cls_imgs_idx   = []
            for i, img_idx in enumerate(range(next_idx, next_idx + len(cls_imgs_path))):
                all_imgs_path.append(cls_imgs_path[i])
                all_imgs_cls.append(str(cls_id))
                cls_imgs_idx.append(img_idx)
            all_cls_imgs.append(cls_imgs_idx)
        return all_imgs_path, all_imgs_cls, all_cls_imgs
    
    all_imgs_path, all_imgs_cls, all_cls_imgs = build_img_lists()

    #Export outputs
    os.makedirs(args.output, exist_ok=True)
    all_cls_imgs = np.array(all_cls_imgs)
    np.save(osp.join(args.output, "class_imgs_ids.npy"), all_cls_imgs)
    
    with open(osp.join(args.output, "img_lst.txt"), "w") as fo:
        fo.write("\n".join(all_imgs_path))

    with open(osp.join(args.output, "img_cls.txt"), "w") as fo:
        fo.write("\n".join(all_imgs_cls))

def build_triplet_pairs(args):
    cls_imgs_ids = np.load(osp.join(args.output, "class_imgs_ids.npy"))
    n_total_cls, n_total_imgs  = cls_imgs_ids.shape 
    simp_triplet_pairs = []
    for trial in range(args.n_trials):
        for pos_cls in range(n_total_cls):
            neg_cls = np.array([i for i in range(n_total_cls) if i != pos_cls])
            pos_imgs = cls_imgs_ids[pos_cls]
            
            for neg_cls_idx in range(len(neg_cls)):
                neg_imgs = cls_imgs_ids[neg_cls[neg_cls_idx]]
                a, p = np.random.choice(pos_imgs, 2)
                n,   = np.random.choice(neg_imgs, 1)
                simp_triplet_pairs.append([a, p, n])
    
    simp_triplet_pairs = np.array(simp_triplet_pairs)
    np.save(osp.join(args.output, args.triplet_out_file), simp_triplet_pairs)
    print(simp_triplet_pairs.shape)
    print(simp_triplet_pairs[:20])

def build_val_set(args):
    data_dir = args.tiny_imgnet_dir
    cls_name_to_id = {}
    with open(osp.join(data_dir, "wnids.txt")) as fi:
        lines = [i.strip() for i in fi.readlines()]
        for i in range(len(lines)):
            cls_name_to_id[lines[i]]= i
    annot_file = osp.join(data_dir, "val", "val_annotations.txt")
    val_annot_df = pd.read_csv(annot_file, delimiter = "\t", header = None)
    with open(osp.join(args.output, "val_lst.txt"), "w") as fo:
        fo.write("\n".join(val_annot_df.iloc[:,0].tolist()))
    val_cls = val_annot_df.iloc[:,1].tolist()
    val_cls = [str(cls_name_to_id[i]) for i in val_cls]
    with open(osp.join(args.output, "val_cls.txt"), "w") as fo:
        fo.write("\n".join(val_cls))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny_imgnet_dir", type=str, default = "/home/hthieu/data/tiny-imagenet-200/" ,
                        help = "directory of the tiny image net folder")
    parser.add_argument("--output", type=str, default = "triplet_pairs",
                        help = "directory of the output folder")
    parser.add_argument("--triplet_out_file", type=str, default = "triplet_199000.npy",
                        help = "directory for the training triplet pairs")
    parser.add_argument("--n_trials", type = int, default = 5,
                        help = "how many times you want to sample triplet pairs from the whole dataset")
    parser.add_argument("--seed", type=int, default = "1606",
                        help = "random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # np.random.seed(args.seed)
    # build_list(args)
    build_triplet_pairs(args)
    # build_val_set(args)
