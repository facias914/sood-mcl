import torch
from mmrotate.models import RotatedFasterRCNN, ROTATED_DETECTORS, RotatedTwoStageDetector
from mmrotate.models.builder import build_loss


@ROTATED_DETECTORS.register_module()
class SemiRotatedFasterRCNN(RotatedFasterRCNN):

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      soft_labels=None,
                      gt_bboxes_ignore=None,
                      get_boxes=False,
                      rcnn_configs=None,
                      loss_configs=None):
        
        super(RotatedTwoStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if not get_boxes:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            if loss_configs != None and loss_configs['type'] == 'FocalLoss':  # for unbiased teacher
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
                self.roi_head.bbox_head.loss_cls = build_loss(loss_configs)
            elif loss_configs == None:
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
            
            if soft_labels != None:
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore)
            else:
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore)
            self.roi_head.bbox_head.loss_cls = roi_cls_loss_orign
            losses.update(roi_losses)
            loss_dict = {}
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_dict[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    loss_dict[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            return loss_dict
        
        with torch.no_grad():
            self.eval()
            assert self.with_bbox, 'Bbox head must be implemented.'
            x = self.extract_feat(img)
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
                                            x, img_metas, proposal_list, rcnn_configs, rescale=False)
            self.train()
        return det_bboxes, det_labels