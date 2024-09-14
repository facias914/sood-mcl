import torch
from mmrotate.models import RotatedRetinaNet, ROTATED_DETECTORS, RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class SemiRotatedRetinaNet(RotatedRetinaNet):

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      get_data=False):

        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        return self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                            gt_labels, gt_bboxes_ignore,
                                            get_data=get_data)