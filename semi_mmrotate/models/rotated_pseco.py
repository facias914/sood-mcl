import torch
from torch.nn import functional as F
import copy
import numpy as np
from mmdet.core import bbox2roi, multi_apply, build_assigner, build_sampler
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
from mmrotate.core.bbox import rbbox_overlaps


@ROTATED_DETECTORS.register_module()
class RotatedPseCo(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss=None, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedPseCo, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.num_classes = train_cfg.get("num_classes", 16)
            self.rcnn_configs = train_cfg.get("rcnn_configs")
            self.rpn_pseudo_threshold = train_cfg.get("rpn_pseudo_threshold")
            self.cls_pseudo_threshold = train_cfg.get("cls_pseudo_threshold")
            self.use_MSL = train_cfg.get("use_MSL")
            self.use_teacher_proposal = train_cfg.get("use_teacher_proposal")
            # initialize assignment to build condidate bags
            self.PLA_iou_thres = self.train_cfg.get("PLA_iou_thres", 0.4)
            initial_assigner_cfg=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=self.PLA_iou_thres,       
                neg_iou_thr=self.PLA_iou_thres,
                min_pos_iou=self.PLA_iou_thres,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1)
            self.initial_assigner = build_assigner(initial_assigner_cfg)
            self.PLA_candidate_topk = self.train_cfg.get("PLA_candidate_topk")
            sampler_cfg=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=False)
            self.sampler = build_sampler(sampler_cfg)
        self.symmetry_aware = symmetry_aware

    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedPseCo, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
            # print(f"{key}: {format_data[key]['img'].shape}")
        losses = dict()
        # supervised forward
        sup_losses = self.forward_sup_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val

        # get student data
        unsup_losses = self.foward_unsup_train(format_data['unsup_weak'], format_data['unsup_strong'])

        if self.iter_count <= self.burn_in_steps:
            unsup_weight = 0
        else: 
            # Train Logic
            # unsupervised forward
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps

        for key, val in unsup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_unsup"] = [unsup_weight * x for x in val]
                else:
                    losses[f"{key}_unsup"] = unsup_weight * val
        
        self.iter_count = self.iter_count + 1

        return losses

    def extract_feat(self, img, model, start_lvl=0):
        """Directly extract features from the backbone+neck."""
        assert start_lvl in [0, 1], \
            f"start level {start_lvl} is not supported."
        x = model.backbone(img)
        # global feature -- [p2, p3, p4, p5, p6, p7]
        if model.with_neck:
            x = model.neck(x)
        if start_lvl == 0:
            return x[:-1]
        elif start_lvl == 1:
            return x[1:]
    
    def forward_sup_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        forward training process for the labeled data.   
        """ 
        losses = dict()
        # high resolution
        x = self.extract_feat(img, self.student, start_lvl=1)
        # RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                              self.student.test_cfg.rpn)
            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RCNN forward and loss 
        roi_losses = self.student.roi_head.forward_train(x, img_metas, proposal_list,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore, gt_masks,
                                                **kwargs)
        losses.update(roi_losses)
       
        return losses
    
    def foward_unsup_train(self, teacher_data, student_data):
        teacher_img = teacher_data["img"]
        student_img = student_data["img"]

        img_metas_teacher = teacher_data["img_metas"]
        img_metas_student = student_data["img_metas"]

        if len(img_metas_student) > 1:
            tnames = [meta["filename"] for meta in img_metas_teacher]
            snames = [meta["filename"] for meta in img_metas_student]
            tidx = [tnames.index(name) for name in snames]
            teacher_img = teacher_img[torch.Tensor(tidx).to(teacher_img.device).long()]
            img_metas_teacher = [img_metas_teacher[idx] for idx in tidx]
        
        # get teacher data
        pseudo_bboxes, pseudo_labels, tea_proposals_tuple = self.extract_teacher_info(
                                teacher_img, img_metas_teacher)
        tea_proposals, tea_feats = tea_proposals_tuple 

        loss = {}
        # RPN stage
        feats = self.extract_feat(student_img, self.student, start_lvl=1)
        stu_rpn_outs, rpn_losses = self.unsup_rpn_loss(  
                feats, pseudo_bboxes, pseudo_labels, img_metas_student)
        loss.update(rpn_losses)

        if self.use_MSL:
            img_ds = F.interpolate(student_img,  # downsampled images
                                scale_factor=0.5, 
                                mode='nearest')
            feats_ds = self.extract_feat(img_ds, self.student, start_lvl=0)
            _, rpn_losses_ds = self.unsup_rpn_loss(feats_ds, 
                                    pseudo_bboxes, pseudo_labels, 
                                    img_metas_student)
            for key, value in rpn_losses_ds.items():
                loss[key + "_V2"] = value 

        # RCNN stage
        """ obtain proposals """
        if self.use_teacher_proposal:
            proposal_list = tea_proposals 
        else :
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *stu_rpn_outs, img_metas_student, cfg=proposal_cfg
            )

        """ obtain teacher predictions for all proposals """
        tea_proposals_copy = copy.deepcopy(tea_proposals)
        with torch.no_grad():
            rois_ = bbox2roi(tea_proposals_copy)
            tea_bbox_results = self.teacher.roi_head._bbox_forward( 
                             tea_feats, rois_)
            
        teacher_infos = {
            "imgs": teacher_img,
            "cls_score": tea_bbox_results["cls_score"][:, :self.num_classes].softmax(dim=-1),
            "bbox_pred": tea_bbox_results["bbox_pred"],
            "feats": tea_feats,  
            "img_metas": img_metas_teacher,
            "proposal_list": tea_proposals_copy} 
        
        rcnn_losses = self.unsup_rcnn_cls_loss(
                            feats, 
                            feats_ds if self.use_MSL else None,  
                            img_metas_student, 
                            proposal_list, 
                            pseudo_bboxes, 
                            pseudo_labels,   
                            teacher_infos=teacher_infos)
        loss.update(rcnn_losses)
        
        return loss
        
    def unsup_rpn_loss(self, stu_feats, pseudo_bboxes, pseudo_labels, img_metas):
        stu_rpn_outs = self.student.rpn_head(stu_feats)
        # rpn loss 
        gt_bboxes_rpn = []
        for bbox, label in zip(pseudo_bboxes, pseudo_labels):
            bbox = bbox[:, :5]
            score = bbox[:, -1]
            pos_inds = torch.nonzero(score >= self.rpn_pseudo_threshold)
            bbox = bbox[pos_inds.reshape(-1)]
            gt_bboxes_rpn.append(bbox) 

        stu_rpn_loss_inputs = stu_rpn_outs + ([bbox.float() for bbox in gt_bboxes_rpn], img_metas)
        rpn_losses = self.student.rpn_head.loss(*stu_rpn_loss_inputs)
        return stu_rpn_outs, rpn_losses
    
    def unsup_rcnn_cls_loss(self,
                        feat,
                        feat_V2,
                        img_metas,
                        proposal_list,
                        pseudo_bboxes,
                        pseudo_labels,
                        teacher_infos=None):  
        gt_bboxes = []
        gt_labels = []
        for bbox, label in zip(pseudo_bboxes, pseudo_labels):
            bbox = bbox[:, :5]
            score = bbox[:, -1]
            pos_inds = torch.nonzero(score >= self.cls_pseudo_threshold)
            bbox = bbox[pos_inds.reshape(-1)]
            label = label[pos_inds.reshape(-1)]
            gt_bboxes.append(bbox) 
            gt_labels.append(label) 

        sampling_results = self.prediction_guided_label_assign(
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    teacher_infos=teacher_infos)
        
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        labels = bbox_targets[0]

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)

        bbox_weights = self.compute_PCV(
                bbox_results["bbox_pred"], 
                labels, 
                selected_bboxes,   
                pos_gt_bboxes_list, 
                pos_assigned_gt_inds_list)
        bbox_weights_ = bbox_weights.pow(2.0)
        pos_inds = (labels >= 0) & (labels < self.student.roi_head.bbox_head.num_classes)
        if pos_inds.any():
            reg_scale_factor = bbox_weights.sum() / bbox_weights_.sum()
        else:
            reg_scale_factor = 0.0

        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *(bbox_targets[:3]),
            bbox_weights_,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = reg_scale_factor * loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0) 
        
        if feat_V2 is not None:
            bbox_results_V2 = self.student.roi_head._bbox_forward(feat_V2, rois)
            loss_V2 = self.student.roi_head.bbox_head.loss(
                bbox_results_V2["cls_score"],
                bbox_results_V2["bbox_pred"],
                rois,
                *(bbox_targets[:3]),
                bbox_weights_,
                reduction_override="none",
            )

            loss["loss_cls_V2"] = loss_V2["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
            loss["loss_bbox_V2"] = reg_scale_factor * loss_V2["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0) 
            if "acc" in loss_V2:
                loss["acc_V2"] = loss_V2["acc"]

        return loss
        
        
    @torch.no_grad()
    def compute_PCV(self, 
                      bbox_preds, 
                      labels, 
                      proposal_list, 
                      pos_gt_bboxes_list, 
                      pos_assigned_gt_inds_list):

        nums = [_.shape[0] for _ in proposal_list]
        labels = labels.split(nums, dim=0)
        bbox_preds = bbox_preds.split(nums, dim=0)
    
        bbox_weights_list = []

        for bbox_pred, label, proposals, pos_gt_bboxes, pos_assigned_gt_inds in zip(
                    bbox_preds, labels, proposal_list, pos_gt_bboxes_list, pos_assigned_gt_inds_list):

            pos_inds = ((label >= 0) & 
                        (label < self.student.roi_head.bbox_head.num_classes)).nonzero().reshape(-1)
            bbox_weights = proposals.new_zeros(bbox_pred.shape[0], 5)
            pos_proposals = proposals[pos_inds]
            if len(pos_inds):
                pos_bbox_weights = proposals.new_zeros(pos_inds.shape[0], 5)
                pos_bbox_pred = bbox_pred[pos_inds]
                decoded_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                        pos_proposals, pos_bbox_pred)
                
                gt_inds_set = torch.unique(pos_assigned_gt_inds)
                
                IoUs = rbbox_overlaps(
                    decoded_bboxes,
                    pos_gt_bboxes,
                    is_aligned=True)
    
                for gt_ind in gt_inds_set:
                    idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
                    if idx_per_gt.shape[0] > 0:
                        pos_bbox_weights[idx_per_gt] = IoUs[idx_per_gt].mean()
                bbox_weights[pos_inds] = pos_bbox_weights
               
            bbox_weights_list.append(bbox_weights)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        return bbox_weights
        
    @torch.no_grad()
    def prediction_guided_label_assign(
                self,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                teacher_infos,
                gt_bboxes_ignore=None,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # get teacher predictions (including cls scores and bbox ious)       
        tea_proposal_list = teacher_infos["proposal_list"] 
        tea_cls_score_concat = teacher_infos["cls_score"]
        tea_bbox_pred_concat = teacher_infos["bbox_pred"]
        num_per_img = [_.shape[0] for _ in tea_proposal_list]
        tea_cls_scores = tea_cls_score_concat.split(num_per_img, dim=0)
        tea_bbox_preds = tea_bbox_pred_concat.split(num_per_img, dim=0)

        decoded_bboxes_list = []
        for bbox_preds, cls_scores, proposals in zip(tea_bbox_preds, tea_cls_scores, tea_proposal_list):
            pred_labels = cls_scores[:, :cls_scores.shape[1]-1].max(dim=-1)[1]
            
            decode_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                        proposals[:, :4], bbox_preds)     
            decoded_bboxes_list.append(decode_bboxes)

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.initial_assigner.assign(  
                decoded_bboxes_list[i],   # construct a candidate bag for each ground truth
                gt_bboxes[i], 
                gt_bboxes_ignore[i], 
                gt_labels[i])
            
            gt_inds = assign_result.gt_inds   
            pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).reshape(-1)

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds] 
            pos_labels = gt_labels[i][pos_assigned_gt_inds]  
            
            tea_pos_cls_score = tea_cls_scores[i][pos_inds]  
           
            tea_pos_bboxes = decoded_bboxes_list[i][pos_inds]
            ious = rbbox_overlaps(tea_pos_bboxes, gt_bboxes[i]) 
            
            wh = proposal_list[i][:, 2:4] - proposal_list[i][:, :2]
            areas = wh[:, 0] * wh[:, 1]
            pos_areas = areas[pos_inds]   
            
            refined_gt_inds = self.assignment_refinement(gt_inds, 
                                       pos_inds, 
                                       pos_assigned_gt_inds, 
                                       ious, 
                                       tea_pos_cls_score, 
                                       pos_areas, 
                                       pos_labels)
    
            assign_result.gt_inds = refined_gt_inds + 1
            sampling_result = self.sampler.sample(
                                assign_result,
                                proposal_list[i],
                                gt_bboxes[i],
                                gt_labels[i])
            sampling_results.append(sampling_result)
        return sampling_results
    
    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds, 
                             ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions 
        # on each image 
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0], ), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)
        
        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]
            target_scores = cls_score[pos_idx_per_gt, target_labels]
            target_areas = areas[pos_idx_per_gt]
            target_IoUs = ious[pos_idx_per_gt, gt_ind]
            
            cost = (target_IoUs * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)
            
            candidate_topk = min(pos_idx_per_gt.shape[0], self.PLA_candidate_topk)   
            topk_ious, _ = torch.topk(target_IoUs, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)      
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[
                target_areas[sort_idx] > 0
            ]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]
            
            refined_pos_gt_inds[pos_idx_per_gt] = pos_assigned_gt_inds[pos_idx_per_gt]
        
        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds
        
    def extract_teacher_info(self, img, img_metas):
        feat = self.extract_feat(img, self.teacher, start_lvl=1)
        proposal_cfg = self.teacher.train_cfg.get(
            "rpn_proposal", self.teacher.test_cfg.rpn
        )   # train_cfg

        rpn_out = list(self.teacher.rpn_head(feat))
        proposal_list = self.teacher.rpn_head.get_bboxes(*rpn_out, img_metas, cfg=proposal_cfg) 

        # teacher proposals
        proposals = copy.deepcopy(proposal_list)

        proposal_list, proposal_label_list = \
            self.teacher.roi_head.simple_test_bboxes( 
            feat, img_metas, proposal_list, 
            self.rcnn_configs, 
            rescale=False
        )   # obtain teacher predictions

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 6) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        det_bboxes = proposal_list

        return det_bboxes, proposal_label_list, \
            (proposals, feat)
    
    def forward_test(self, imgs, img_metas, **kwargs):

        return super(RotatedSemiDetector, self).forward_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        
        model = self.model(**kwargs)
        assert model.with_bbox, 'Bbox head must be implemented.'
        
        x = self.extract_feat(img, model, start_lvl=1)

        if proposals is None:
            proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return model.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

