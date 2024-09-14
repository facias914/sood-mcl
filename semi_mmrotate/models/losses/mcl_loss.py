import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply
from mmrotate.models import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class RotatedMCLLoss(nn.Module):
    def __init__(self, cls_channels=16):
        super(RotatedMCLLoss, self).__init__()
        self.cls_channels = cls_channels
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')

    def pre_processing(self, logits, alone_angle=True):
        if alone_angle:
            cls_scores, bbox_preds, angle_preds, centernesses = logits
            assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        else:
            cls_scores, bbox_preds, centernesses = logits
            assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   
        img_logits_list = []
        for img_id in range(batch_size):
            img_logits = []
            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, self.cls_channels) for x in cls_scores
            ], dim=0))

            if alone_angle:
                img_logits.append(torch.cat([
                    torch.cat([x[img_id], y[img_id]], dim=0).permute(1, 2, 0).reshape(-1, 5) for x, y in
                    zip(bbox_preds, angle_preds)
                ], dim=0))
            else:
                img_logits.append(torch.cat([
                    x[img_id].permute(1, 2, 0).reshape(-1, 5) for x in bbox_preds
                ], dim=0))

            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, 1) for x in centernesses
            ], dim=0))

            img_logits_list.append(img_logits)

        return img_logits_list

    def loss_single(self, t_logits_list, s_logits_list, level_inds):
        t_cls_scores, t_bbox_preds, t_centernesses = tuple(t_logits_list)
        s_cls_scores, s_bbox_preds, s_centernesses = tuple(s_logits_list)
        with torch.no_grad():
            teacher_probs = t_cls_scores.sigmoid()
            teacher_bboxes = t_bbox_preds
            teacher_centernesses = t_centernesses.sigmoid()

            # P3
            teacher_probs_p3 = teacher_probs[:level_inds[0]]
            teacher_centernesses_p3 = teacher_centernesses[:level_inds[0]]
            joint_confidences_p3 = teacher_probs_p3 * teacher_centernesses_p3
            max_vals_p3 = torch.max(joint_confidences_p3, 1)[0]
            selected_inds_p3 = torch.topk(max_vals_p3, joint_confidences_p3.size(0))[1][:2000]

            # P4
            teacher_probs_p4 = teacher_probs[level_inds[0]:level_inds[1]]
            teacher_centernesses_p4 = teacher_centernesses[level_inds[0]:level_inds[1]]
            joint_confidences_p4 = teacher_probs_p4 * teacher_centernesses_p4
            select_inds_p4 = torch.arange(level_inds[0], level_inds[1]).to(joint_confidences_p4.device)
            max_vals_p4 = torch.max(joint_confidences_p4, 1)[0]
            selected_inds_p4 = select_inds_p4[torch.topk(max_vals_p4, joint_confidences_p4.size(0))[1][:2000]]

            # P5, P6, P7
            confidences_rest = teacher_probs[level_inds[1]:]
            selected_inds_rest = torch.arange(level_inds[1], teacher_probs.shape[0]).to(confidences_rest.device)

            # coarse_inds
            selected_inds_coarse = torch.cat([selected_inds_p3, selected_inds_p4, selected_inds_rest], 0)
            
            # fine_inds
            all_confidences = torch.cat([joint_confidences_p3, joint_confidences_p4, confidences_rest], 0)
            max_vals = torch.max(all_confidences, 1)[0]
            selected_inds = torch.nonzero(max_vals > 0.02).squeeze(-1)

            selected_inds, counts = torch.cat([selected_inds_coarse, selected_inds], 0).unique(return_counts=True)
            selected_inds = selected_inds[counts>1]

            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds]
            b_mask = weight_mask > 0.

        if b_mask.sum() == 0:
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=max_vals,
                reduction="sum",
            ) / max_vals.sum()
            loss_bbox = torch.zeros(1).squeeze().to(max_vals.device)
            loss_centerness = torch.zeros(1).squeeze().to(max_vals.device)
        else:
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=weight_mask,
                reduction="sum",
            ) / weight_mask.sum()

            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                teacher_bboxes[b_mask],
            ) * weight_mask[:, None][b_mask]).mean() * 10

            loss_centerness = (F.binary_cross_entropy(
                s_centernesses[b_mask].sigmoid(),
                teacher_centernesses[b_mask],
                reduction='none'
            )* weight_mask[:, None][b_mask]).mean() * 10

        return loss_cls, loss_bbox, loss_centerness

    def forward(self, teacher_logits, student_logits, img_metas=None, alone_angle=True):

        img_t_logits_list = self.pre_processing(teacher_logits, alone_angle)
        img_s_logits_list = self.pre_processing(student_logits, alone_angle)

        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_logits[0]]
        level_inds = []
        start = 0
        for size in featmap_sizes:
            start = start + size[0] * size[0]
            level_inds.append(start)
        level_inds = level_inds[:2]

        # get labels and bbox_targets of each image
        losses_list = multi_apply(
                            self.loss_single,
                            img_t_logits_list,
                            img_s_logits_list,
                            level_inds=level_inds)

        unsup_losses = dict(
            loss_cls=sum(losses_list[0]) / len(losses_list[0]),
            loss_bbox=sum(losses_list[1]) / len(losses_list[1]),
            loss_centerness=sum(losses_list[2]) / len(losses_list[2])
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0 
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape) 
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta) 
    pos = weight > 0

    # positive goes to teacher quality 
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos] 
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss