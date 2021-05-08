import torch 
import ipdb
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

def img_ranking_evaluate_Market1501(
        gal_embs, gal_lbls, que_embs, que_lbls, export_result = False):
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

if __name__ == "__main__":
    torch.manual_seed(10)
    n_imgs = 50
    img_embs = torch.randn(n_imgs, 128)
    labels   = torch.randint(0,50,(n_imgs,1), dtype=torch.int)
    img_ranking_evaluate(img_embs, labels)
    # a = torch.tensor([[0,1,2],[1,1,2]])
    # b = torch.tensor([[0],[1]])
    # print(a.shape)
    # print(b.shape)
    # print(a==b)
