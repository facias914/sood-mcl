import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder
from mmcv.ops import box_iou_rotated
from mmdet.core.anchor.point_generator import MlvlPointGenerator


@ROTATED_LOSSES.register_module()
class RotatedARSLLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedARSLLoss, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.fpn_stride=[8, 16, 32, 64, 128]
        self.prior_generator = MlvlPointGenerator(self.fpn_stride)
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator(self.fpn_stride)
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.loss_type = loss_type

    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1).view(-1, 5)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in centernesses
        ], dim=1).view(-1, 1)
        return cls_scores, bbox_preds, centernesses
    
    def inside_bbox_mask(self, points, gt_bboxes):
        num_points = points.size(0)  
        num_gts = gt_bboxes.size(0)  
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        return inside_gt_bbox_mask
    
    # select potential positives from hard negatives 
    def hard_neg_mining(self,
                        cls_score,
                        loc_ltrba,
                        quality,
                        pos_ind,
                        hard_neg_ind,
                        loc_mask,
                        loc_targets,
                        iou_thresh=0.6):
        # get points locations and strides
        scale_list = []
        scale = [0, 1, 2, 3, 4]
        points_list = self.prior_generator.grid_priors(
                self.lvl_hw,
                dtype=loc_ltrba.dtype,
                device=loc_ltrba.device)
        for fpn_scale, HW in zip(scale, self.lvl_hw):
            h, w = HW
            lvl_scales = torch.full([h * w, 1], fpn_scale)
            scale_list.append(lvl_scales)
        points = torch.cat(points_list, axis=0)
        scales = torch.cat(scale_list, axis=0)

        # cls scores
        joint_confidence = F.sigmoid(cls_score) * F.sigmoid(quality)
        max_vals, class_ind = torch.max(joint_confidence, axis=-1)

        ### calculate iou between positive and hard negative
        # decode pos bbox
        pos_cls = max_vals[pos_ind]
        pos_loc = loc_ltrba[pos_ind]
        pos_points = points[pos_ind]
        pos_loc = pos_loc
        pos_bbox = self.bbox_coder.decode(pos_points, pos_loc)
        pos_scales = scales[pos_ind]
    
        # decode hard negative bbox
        hard_neg_loc = loc_ltrba[hard_neg_ind]
        hard_neg_points = points[hard_neg_ind]
        hard_neg_loc = hard_neg_loc
        hard_neg_bbox = self.bbox_coder.decode(hard_neg_points, hard_neg_loc)
        hard_neg_scales = scales[hard_neg_ind]
        # iou between pos bbox and hard negative bbox
        hard_neg_pos_iou = box_iou_rotated(hard_neg_bbox, pos_bbox)

        ### select potential positives from hard negatives
        # scale flag
        scale_temp = torch.abs(
            pos_scales.reshape([-1])[None, :] - hard_neg_scales.reshape([-1])
            [:, None])
        scale_flag = (scale_temp <= 1.).to(hard_neg_pos_iou.device)
        # iou flag
        iou_flag = (hard_neg_pos_iou >= iou_thresh)
        # same class flag
        pos_class = class_ind[pos_ind]
        hard_neg_class = class_ind[hard_neg_ind]
        class_flag = pos_class[None, :] - hard_neg_class[:, None]
        class_flag = (class_flag == 0)
        # hard negative point inside positive bbox flag
        inside_flag = self.inside_bbox_mask(hard_neg_points, pos_bbox)
        # reset iou
        valid_flag = (iou_flag & class_flag & inside_flag & scale_flag)
        invalid_iou = torch.zeros_like(hard_neg_pos_iou)
        hard_neg_pos_iou = torch.where(valid_flag, hard_neg_pos_iou,
                                        invalid_iou)
        pos_hard_neg_max_iou = hard_neg_pos_iou.max(axis=-1)[0]
        # selece potential pos
        potential_pos_ind = (pos_hard_neg_max_iou > 0.)
        num_potential_pos = torch.nonzero(potential_pos_ind).shape[0]
        if num_potential_pos == 0:
            return None

        ### calculate loc target：aggregate all matching bboxes as the bbox targets of potential pos
        # prepare data
        potential_points = hard_neg_points[potential_pos_ind]
        potential_valid_flag = valid_flag[potential_pos_ind]
        potential_pos_ind = hard_neg_ind[potential_pos_ind]

        # get cls and box of matching positives
        pos_cls = max_vals[pos_ind]
        expand_pos_bbox = pos_bbox.expand(num_potential_pos, pos_bbox.shape[0], pos_bbox.shape[1])
        expand_pos_cls = pos_cls.expand(num_potential_pos, pos_cls.shape[0])
        invalid_cls = torch.zeros_like(expand_pos_cls)
        expand_pos_cls = torch.where(potential_valid_flag, expand_pos_cls,
                                      invalid_cls)
        expand_pos_cls = torch.unsqueeze(expand_pos_cls, axis=-1)
        # aggregate box based on cls_score
        agg_bbox = (expand_pos_bbox * expand_pos_cls).sum(axis=1) \
            / expand_pos_cls.sum(axis=1)
        agg_ltrba = self.bbox_coder.encode(potential_points, agg_bbox)

        # loc target for all pos
        loc_targets[potential_pos_ind] = agg_ltrba
        loc_mask[potential_pos_ind] = 1.

        return loc_mask, loc_targets

    def forward(self, teacher_logits, student_logits, img_metas=None, **kwargs):

        self.lvl_hw = []
        for t in teacher_logits[0]:
            _, _, H, W = t.shape
            self.lvl_hw.append([H, W])

        t_cls_scores, t_bbox_preds, t_iou_preds = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_iou_preds = self.convert_shape(student_logits)

        with torch.no_grad():
            ### sample selection
            # prepare datas
            joint_confidence = t_cls_scores.sigmoid() * t_iou_preds.sigmoid()
            max_vals, class_ind = torch.max(joint_confidence, axis=-1)
            cls_mask = torch.zeros_like(max_vals)
            num_pos, num_hard_neg = 0, 0

            # mean-std selection
            candidate_ind = torch.nonzero(max_vals >= 0.1).squeeze(axis=-1)
            num_candidate = candidate_ind.shape[0]
            if num_candidate > 0:
                # pos thresh = mean + std to select pos samples
                candidate_score = max_vals[candidate_ind]
                candidate_score_mean = candidate_score.mean()
                candidate_score_std = candidate_score.std()
                pos_thresh = (candidate_score_mean + candidate_score_std).clip(
                    max=0.4)
                # select pos
                pos_ind = torch.nonzero(max_vals >= pos_thresh).squeeze(axis=-1)
                num_pos = pos_ind.shape[0]
                # select hard negatives as potential pos
                hard_neg_ind = (max_vals >= 0.1) & (max_vals < pos_thresh)
                hard_neg_ind = torch.nonzero(hard_neg_ind).squeeze(axis=-1)
                num_hard_neg = hard_neg_ind.shape[0]
            # if not positive, directly select top-10 as pos.
            if (num_pos == 0):
                num_pos = 10
                _, pos_ind = torch.topk(max_vals, k=num_pos)
            cls_mask[pos_ind] = 1.

            ### Consistency Regularization Training targets
            # cls targets
            pos_class_ind = class_ind[pos_ind]
            cls_targets = torch.zeros_like(t_cls_scores)
            cls_targets[pos_ind] = t_cls_scores.sigmoid()[pos_ind]  # differ with paper, for distilling more information from teacher

            # hard negative cls target
            if num_hard_neg != 0:
                cls_targets[hard_neg_ind] = t_cls_scores.sigmoid()[hard_neg_ind]
            # loc targets
            loc_targets = torch.zeros_like(t_bbox_preds)
            loc_targets[pos_ind] = t_bbox_preds[pos_ind]
            # iou targets
            iou_targets = torch.zeros_like(t_iou_preds).squeeze(1)
            iou_targets[pos_ind] = F.sigmoid(
                torch.squeeze(
                    t_iou_preds, axis=-1)[pos_ind])

            loc_mask = cls_mask.clone()
            # select potential positive from hard negatives for loc_task training
            if (num_hard_neg > 0):
                results = self.hard_neg_mining(t_cls_scores, t_bbox_preds, t_iou_preds, pos_ind,
                                            hard_neg_ind, loc_mask, loc_targets)
                if results is not None:
                    loc_mask, loc_targets = results
                    loc_pos_ind = torch.nonzero(loc_mask > 0.).squeeze(axis=-1)
                    iou_targets[loc_pos_ind] = F.sigmoid(
                        torch.squeeze(
                            t_iou_preds, axis=-1)[loc_pos_ind])
            
            ### Training Weights and avg factor
            # find positives
            cls_pos_ind = torch.nonzero(cls_mask > 0.).squeeze(axis=-1)
            loc_pos_ind = torch.nonzero(loc_mask > 0.).squeeze(axis=-1)
            # cls weight
            cls_sample_weights = torch.ones([cls_targets.shape[0]])
            cls_avg_factor = torch.max(cls_targets[cls_pos_ind],
                                        axis=-1)[0].sum()     # differ with paper
            # loc weight
            loc_sample_weights = torch.max(cls_targets[loc_pos_ind], axis=-1)[0]
            loc_avg_factor = loc_sample_weights.sum()
            # iou weight
            iou_sample_weights = torch.ones([loc_pos_ind.shape[0]])
            iou_avg_factor = torch.tensor(loc_pos_ind.size(0))

        ### unsupervised loss
        # cls loss
        loss_cls = self.quality_focal_loss(
            s_cls_scores.sigmoid(),
            cls_targets,
            quality=s_iou_preds.sigmoid(),
            weights=cls_sample_weights,
            avg_factor=cls_avg_factor)
        # iou loss
        pos_stu_iou = s_iou_preds.sigmoid().squeeze(-1)[loc_pos_ind]
        pos_iou_targets = iou_targets[loc_pos_ind]
        loss_iou = F.binary_cross_entropy(
            pos_stu_iou, pos_iou_targets,
            reduction='none') * iou_sample_weights.to(pos_stu_iou.device)
        loss_iou = loss_iou.sum() / iou_avg_factor.to(pos_stu_iou.device)
        # box loss
        pos_stu_loc = s_bbox_preds[loc_pos_ind]
        pos_loc_targets = loc_targets[loc_pos_ind]

        loss_bbox = (self.bbox_loss(
                pos_stu_loc,
                pos_loc_targets,
            ) * loc_sample_weights.to(pos_stu_loc.device)[:, None]).sum() / loc_avg_factor.to(pos_stu_loc.device)

        loss_all = {
            "loss_cls": loss_cls,
            "loss_box": loss_bbox,
            "loss_iou": loss_iou,
        }
        return loss_all
    
    # cls loss: iou-based soft lable with joint iou
    def quality_focal_loss(self,
                           stu_cls,
                           targets,
                           quality=None,
                           weights=None,
                           alpha=0.75,
                           gamma=2.0,
                           avg_factor='sum'):
        if quality is not None:
            stu_cls = stu_cls * quality

        focal_weight = (stu_cls - targets).abs().pow(gamma) * (targets > 0.0) + \
            alpha * (stu_cls - targets).abs().pow(gamma) * \
            (targets <= 0.0)

        loss = F.binary_cross_entropy(
            stu_cls, targets, reduction='none') * focal_weight

        if weights is not None:
            loss = loss * weights.reshape([-1, 1]).to(loss.device)
        if avg_factor is not None:
            loss = loss.sum() / avg_factor.to(loss.device)
        return loss