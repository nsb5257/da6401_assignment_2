import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset
from inference import MetricsCalculator

class WandBLogger:
    @staticmethod
    def log_activations(epoch, features):
        if 'block3' in features:
            activations = features['block3'].detach().cpu().numpy().flatten()
            wandb.log({"Conv Block 3 Activations": wandb.Histogram(activations), "epoch": epoch})

    @staticmethod
    def log_feature_maps(epoch, features):
        for block_name in ['block1', 'block5']:
            if block_name in features:
                fmap = features[block_name][0].detach().cpu()
                imgs = []
                for c in range(min(8, fmap.shape[0])):
                    f = fmap[c].numpy()
                    f_min, f_max = f.min(), f.max()
                    f_norm = (f - f_min) / (f_max - f_min + 1e-5)
                    imgs.append(wandb.Image(f_norm, caption=f"{block_name}_ch{c}"))
                wandb.log({f"Feature Maps/{block_name}": imgs, "epoch": epoch})

    @staticmethod
    def log_bounding_boxes(images, preds, targets, cls_logits, epoch):
        table = wandb.Table(columns=["ID", "Confidence", "IoU", "BBox Image"])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
        vis_images = torch.clamp(images * std + mean, 0, 1)

        for i in range(min(10, images.size(0))):
            img_np = (vis_images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            p_box = preds[i].cpu().detach().numpy()
            t_box = targets[i].cpu().detach().numpy()

            def to_xyxy(b):
                return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2

            px1, py1, px2, py2 = to_xyxy(p_box)
            tx1, ty1, tx2, ty2 = to_xyxy(t_box)
            inter = max(0, min(px2,tx2)-max(px1,tx1)) * max(0, min(py2,ty2)-max(py1,ty1))
            union = p_box[2]*p_box[3] + t_box[2]*t_box[3] - inter
            iou = float(inter / (union + 1e-6))
            conf = float(F.softmax(cls_logits[i].detach(), dim=0).max().item())

            def to_corners(box):
                x_c, y_c, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                return {"minX": max(0., x_c-w/2), "maxX": min(224., x_c+w/2),
                        "minY": max(0., y_c-h/2), "maxY": min(224., y_c+h/2)}

            wandb_boxes = {
                "predictions": {"box_data": [{"position": to_corners(p_box), "class_id": 1, "domain": "pixel"}], "class_labels": {1: "Prediction"}},
                "ground_truth": {"box_data": [{"position": to_corners(t_box), "class_id": 2, "domain": "pixel"}], "class_labels": {2: "Ground Truth"}}
            }
            table.add_data(f"Img_{i}", round(conf, 3), round(iou, 3), wandb.Image(img_np, boxes=wandb_boxes))
        wandb.log({f"BBox Evaluation - Epoch {epoch}": table})

def apply_transfer_strategy(model, strategy):
    if strategy == "strict_extractor":
        for param in model.shared_encoder.parameters(): param.requires_grad = False
    elif strategy == "partial_fine_tuning":
        for param in model.shared_encoder.parameters(): param.requires_grad = False
        for param in model.shared_encoder.block4.parameters(): param.requires_grad = True
        for param in model.shared_encoder.block5.parameters(): param.requires_grad = True

def train_pipeline(config):
    wandb.init(project="DA6401-Assignment2", name=config.get("run_name"), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = MultiTaskPerceptionModel(classifier_path=config.get('classifier_path', ''), 
                                    localizer_path=config.get('localizer_path', ''),
                                    unet_path=config.get('unet_path', ''), use_bn=config.get('use_bn', True))
    
    if "dropout_p" in config:
        for module in model.modules():
            if type(module).__name__ == "CustomDropout": module.p = config["dropout_p"]

    apply_transfer_strategy(model, config['transfer_strategy'])
    model = model.to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    criteria = {'cls': nn.CrossEntropyLoss(), 'loc': nn.MSELoss(), 'iou': IoULoss(), 'seg': nn.CrossEntropyLoss()}

    KAGGLE_ROOT = os.path.abspath("./data")
    full_ds = OxfordIIITPetDataset(split="train", data_dir=KAGGLE_ROOT)
    train_ds, val_ds = random_split(full_ds, [int(0.9*len(full_ds)), len(full_ds)-int(0.9*len(full_ds))], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            imgs, lbls, bboxes, masks = batch['image'].to(device), batch['label'].to(device), batch['bbox'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            
            l_cls = criteria['cls'](out['classification'], lbls)
            l_loc = criteria['loc'](out['localization'], bboxes) + criteria['iou'](out['localization'], bboxes)
            l_seg = criteria['seg'](out['segmentation'], masks)
            
            loss = (config.get('lambda_cls', 1.0) * l_cls) + \
           (config['lambda_loc'] * l_loc) + \
           l_seg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            wandb.log({
                "Train/Batch_Total_Loss": loss.item(),
                "Train/Batch_Class_Loss": l_cls.item(),
                "Train/Batch_Loc_Loss": l_loc.item(),
                "Train/Batch_Seg_Loss": l_seg.item()
            })

        model.eval()
        v_dice, v_pix, v_f1, v_loss = 0, 0, 0, 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_imgs, v_lbls, v_bboxes, v_masks = v_batch['image'].to(device), v_batch['label'].to(device), v_batch['bbox'].to(device), v_batch['mask'].to(device)
                v_out = model(v_imgs)
                v_loss += (criteria['cls'](v_out['classification'], v_lbls) + \
                          (config['lambda_loc'] * (criteria['loc'](v_out['localization'], v_bboxes) + criteria['iou'](v_out['localization'], v_bboxes))) + \
                          criteria['seg'](v_out['segmentation'], v_masks))
                v_dice += MetricsCalculator.calculate_dice_score(v_out['segmentation'], v_masks)
                v_pix += MetricsCalculator.pixel_accuracy(v_out['segmentation'], v_masks)
                v_f1 += MetricsCalculator.calculate_macro_f1(v_out['classification'], v_lbls)

        avg_val_loss = v_loss / len(val_loader)
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"✅ Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {v_f1/len(val_loader):.4f}")

        wandb.log({
            "Train/Epoch_Loss": avg_train_loss,
            "Val/Epoch_Loss": avg_val_loss,
            "Val/Dice_Score": v_dice / len(val_loader),
            "Val/Pixel_Accuracy": v_pix / len(val_loader),
            "Val/Macro_F1": v_f1 / len(val_loader),
            "epoch": epoch
        })
        
        # Telemetry Visuals
        with torch.no_grad():
            s_batch = next(iter(val_loader))
            s_imgs, s_out = s_batch['image'].to(device), model(s_batch['image'].to(device))
            base = model.module if isinstance(model, nn.DataParallel) else model
            _, feats = base.shared_encoder(s_imgs, return_features=True)
            WandBLogger.log_activations(epoch, feats)
            WandBLogger.log_feature_maps(epoch, feats)
            WandBLogger.log_bounding_boxes(s_imgs, s_out['localization'], s_batch['bbox'].to(device), s_out['classification'], epoch)

    os.makedirs("checkpoints", exist_ok=True)
    base = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"state_dict": base.classifier_model.state_dict()}, "checkpoints/classifier.pth")
    torch.save({"state_dict": base.localizer_model.state_dict()}, "checkpoints/localizer.pth")
    torch.save({"state_dict": base.unet_model.state_dict()}, "checkpoints/unet.pth")
    wandb.finish()
