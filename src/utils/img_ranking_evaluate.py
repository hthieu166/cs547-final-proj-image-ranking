import torch 
from sklearn.metrics import average_precision_score
import ipdb
import numpy as np

def img_ranking_evaluate_tinyImageNet(img_embs, labels, export_result = False):
    #Pair-wise L2 distance between each pair
    c_dist   = torch.cdist(img_embs, img_embs)
    #Sort the image id in ascending order by its distance 
    img_rank = torch.argsort(c_dist, dim = 1)
    img_label_pred = (labels[img_rank]).squeeze() #shape: N_img x N_img
    labels         = labels.unsqueeze(1) #shape: N_img x 1
    #Calculate top K-accuracy:
    correct_pred  = img_label_pred == labels
    top_1  = correct_pred[:, 1]
    top_10 = correct_pred[:, 1:11]
    top_49 = correct_pred[:, 1:50]
    top_1  = top_1.to(torch.float).mean()
    top_10 = top_10.to(torch.float).mean()
    top_49 = top_49.to(torch.float).mean()
    ret_res= {
        "acc_top1": top_1.cpu().item(),
        "acc_top10": top_10.cpu().item(),
        "acc_top49": top_49.cpu().item()
    }
    if (export_result):
        ret_res["preds"] = img_rank.cpu().numpy()
    return ret_res

def calculate_mAP(dist, match):
    aps = []
    scores = 1 / (1 + dist)
    for i in range(len(dist)):
        ap = average_precision_score(match[i], scores[i])
        aps.append(ap)
    return np.mean(aps) 


def img_ranking_evaluate_Market1501(
        gal_embs, gal_lbls, que_embs, que_lbls, export_result = False):
    
    #Pair-wise L2 distance between each pair
    c_dist   = torch.cdist(que_embs, gal_embs)
    
    #Sort the image id in ascending order by its distance 
    img_rank = torch.argsort(c_dist, dim = 1)
    img_label_pred = (gal_lbls[img_rank]).squeeze() #shape: N_img x N_img
    labels         = que_lbls.unsqueeze(1) #shape: N_img x 1
    
    #Caluclate mAP
    meanAP = calculate_mAP(c_dist.cpu().detach().numpy(), 
                        (gal_lbls == labels).cpu().detach().numpy())

    #Calculate top K-accuracy:
    correct_pred  = img_label_pred == labels
    top_1  = correct_pred[:, 1]
    top_5  = correct_pred[:, 1:6]
    top_10 = correct_pred[:, 1:11]
    top_49 = correct_pred[:, 1:50]
    top_1  = top_1.to(torch.float).mean()
    top_10 = top_10.to(torch.float).mean()
    top_49 = top_49.to(torch.float).mean()
    top_5  = top_5.to(torch.float).mean()
    ret_res= {
        "acc_top1": top_1.cpu().item(),
        "acc_top5": top_5.cpu().item(),
        "acc_top10": top_10.cpu().item(),
        "acc_top49": top_49.cpu().item(),
        "mAP": meanAP
    }
    if (export_result):
        ret_res["preds"] = img_rank.cpu().numpy()
    return ret_res

def compute_map(correct_preds):
    correct_preds.cumsum()
if __name__ == "__main__":
    torch.manual_seed(10)
    n_imgs = 50
    # img_embs = torch.randn(n_imgs, 128)
    # labels   = torch.randint(0,10,(n_imgs,1), dtype=torch.int)
    # img_ranking_evaluate(img_embs, labels)
    gal_embs = torch.randn(n_imgs, 128)
    que_embs = torch.randn(n_imgs//2, 128)
    gal_lbls = torch.randint(0,10,(n_imgs,1), dtype=torch.int)
    que_lbls = torch.randint(0,10,(n_imgs//2,1), dtype=torch.int)
    print(img_ranking_evaluate_Market1501(gal_embs, gal_lbls, que_embs, que_lbls))