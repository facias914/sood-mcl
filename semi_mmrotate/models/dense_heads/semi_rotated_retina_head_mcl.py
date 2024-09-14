# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean, images_to_levels

from mmrotate.models import ROTATED_HEADS, build_loss
from mmrotate.models.dense_heads import RotatedRetinaHead
from mmrotate.core import multiclass_nms_rotated

INF = 1e8

@ROTATED_HEADS.register_module()
class SemiRotatedRetinaHeadMCL(RotatedRetinaHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                loss_centerness=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF))
        super(SemiRotatedRetinaHeadMCL, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.center_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      get_data=False,
                      **kwargs):
        if get_data:
            return self(x)
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        center_pred = self.center_reg(reg_feat)
        return cls_score, bbox_pred, center_pred
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'center_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             center_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_center_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_center_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, centerness_targets_list) = cls_reg_center_targets

        # cat all level and all batch predicted confidences and gt
        level_cls_scores_list = [cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        all_cls_scores = torch.cat(level_cls_scores_list, 0)
        level_bbox_preds_list = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5) for bbox_pred in bbox_preds]
        all_bbox_preds = torch.cat(level_bbox_preds_list, 0)
        level_center_preds_list = [center_pred.permute(0, 2, 3, 1).reshape(-1) for center_pred in center_preds]
        all_center_preds = torch.cat(level_center_preds_list, 0)

        level_labels_list = [label.reshape(-1) for label in labels_list]
        all_labels = torch.cat(level_labels_list, 0)
        level_labels_weights_list = [label_weight.reshape(-1) for label_weight in label_weights_list]
        all_labels_weights = torch.cat(level_labels_weights_list, 0)
        level_bbox_targets_list = [bbox_target.reshape(-1, 5) for bbox_target in bbox_targets_list]
        all_bbox_targets = torch.cat(level_bbox_targets_list, 0)
        level_centerness_targets_list = [centerness_target.reshape(-1) for centerness_target in centerness_targets_list]
        all_centerness_targets = torch.cat(level_centerness_targets_list, 0)

        pos_inds = (all_labels != self.num_classes)
        num_pos = (all_labels != self.num_classes).sum()
        
        # centerness loss
        losses_centerness = self.loss_centerness(
            all_center_preds[pos_inds], all_centerness_targets[pos_inds], avg_factor=num_pos)
        centerness_denorm = max(
                reduce_mean(all_centerness_targets[pos_inds].sum().detach()), 1e-6)
        
        # classification loss
        joint_confidence_scores = all_cls_scores.sigmoid() * all_center_preds.sigmoid()[:, None]
        losses_cls = self.loss_cls(
                    joint_confidence_scores, (all_labels, all_centerness_targets), avg_factor=num_pos)
            
        # regression loss
        losses_bbox = self.loss_bbox(
            all_bbox_preds[pos_inds],
            all_bbox_targets[pos_inds],
            weight=all_centerness_targets[pos_inds][:, None],
            avg_factor=centerness_denorm)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_center=losses_centerness)
    
    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # concat all level anchors to a single tensor
        level_anchor_list = []
        level_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            level_anchor_list.append(anchor_list[i])
            level_valid_flag_list.append(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            level_anchor_list,
            level_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas)
        (all_labels, all_label_weights, all_bbox_targets, all_centerness_targets) = results[:4]

        # split targets to a list w.r.t. multiple levels
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        centerness_targets_list = images_to_levels(all_centerness_targets,
                                             num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list, centerness_targets_list)

        return res
    
    def _get_targets_single(self,
                            anchors_list,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta):
        orign_gt_bboxes = gt_bboxes
        level_anchors = anchors_list
        cat_anchors = torch.cat(anchors_list, 0)
        num_levels = len(level_anchors)
        num_gts = len(gt_bboxes)

        center_points = []
        for level_anchor in level_anchors:
            point = level_anchor[:, :2]
            center_points.append(point)

        expanded_regress_ranges = [
            center_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                center_points[i]) for i in range(num_levels)
        ]

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(center_points, dim=0)

        # the number of points per img, per lvl
        num_points = len(concat_points)

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        areas = areas[None].repeat(num_points, 1)
        concat_regress_ranges = concat_regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        concat_points = concat_points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = concat_points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)
        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        max_regress_distance = torch.stack((left, top, right, bottom), -1).max(-1)[0]

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]

        gaussian_center = offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = gaussian_center < 1

        # condition2: limit the regression range for each location
        inside_regress_range = (
            (max_regress_distance >= concat_regress_ranges[..., 0])
            & (max_regress_distance <= concat_regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        label_weights = concat_points.new_ones(num_points, dtype=torch.float)

        centerness_targets = 1 - gaussian_center[range(num_points), min_area_inds]

        bbox_targets = torch.zeros_like(cat_anchors)
        pos_inds = (labels != self.num_classes)
        pos_bboxes = cat_anchors[pos_inds]
        pos_gt_bboxes = orign_gt_bboxes[min_area_inds[min_area != INF]]

        scale = ((pos_gt_bboxes[:, 2] * pos_gt_bboxes[:, 3]) / (1024 * 1024)).pow(0.2)
        centerness_targets[pos_inds] = centerness_targets[pos_inds] ** scale

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

        return (labels, label_weights, bbox_targets, centerness_targets)
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centerness_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centerness_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(centerness_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centerness_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    centerness_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    centerness_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           centerness_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(centerness_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(cls_score_list,
                                                 bbox_pred_list, centerness_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == centerness.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels