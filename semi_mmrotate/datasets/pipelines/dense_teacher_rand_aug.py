import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
import copy
import pycocotools.mask as maskUtils
import os.path as osp
import mmcv
from mmdet.core import BitmapMasks, PolygonMasks
from mmcv.parallel import DataContainer as DC
from collections.abc import Sequence
import torch

from mmcv import is_list_of
from mmdet.datasets.pipelines import Compose as BaseCompose

from mmrotate.datasets.builder import ROTATED_PIPELINES
from mmrotate.datasets.pipelines import PolyRandomRotate

def to_tensor(data):

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class DTSingleOperation:
    def __init__(self):
        self.transform = None

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = self.transform(results[key])
        return results


@ROTATED_PIPELINES.register_module()
class DTToPILImage(DTSingleOperation):
    def __init__(self):
        super(DTToPILImage, self).__init__()
        self.transform = transforms.ToPILImage()


# DT represents for Dense Teacher
@ROTATED_PIPELINES.register_module()
class DTRandomApply:
    def __init__(self, operations, p=0.5):
        self.p = p
        if is_list_of(operations, dict):
            self.operations = []
            for ope in operations:
                self.operations.append(build_dt_aug(**ope))
        else:
            self.operations = operations

    def __call__(self, results):
        if self.p < np.random.random():
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            for ope in self.operations:
                img = ope(img)
            results[key] = img
        return results


@ROTATED_PIPELINES.register_module()
class DTRandomGrayscale(DTSingleOperation):
    def __init__(self, p=0.2):
        super(DTRandomGrayscale, self).__init__()
        self.transform = transforms.RandomGrayscale(p=p)


@ROTATED_PIPELINES.register_module()
class DTRandCrop(DTSingleOperation):
    def __init__(self):
        super(DTRandCrop, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
            transforms.ToPILImage(),
        ])


@ROTATED_PIPELINES.register_module()
class DTToNumpy:
    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = np.asarray(results[key])
        return results


# ST represents Soft Teacher
@ROTATED_PIPELINES.register_module()
class STMultiBranch(object):
    def __init__(self, is_seq=False, **transform_group):
        self.is_seq = is_seq
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        if self.is_seq:
            weak_pipe = self.transform_group['unsup_weak']
            strong_pipe = self.transform_group['unsup_strong']
            res = weak_pipe(copy.deepcopy(results))
            multi_results.append(copy.deepcopy(res))
            res.pop('tag')
            multi_results.append(strong_pipe(res))
            for k, v in self.transform_group.items():
                if 'common' in k:
                    for idx in range(len(multi_results)):
                        multi_results[idx] = v(multi_results[idx])
        else:
            for k, v in self.transform_group.items():
                res = v(copy.deepcopy(results))
                if res is None:
                    return None
                multi_results.append(res)
        return multi_results


@ROTATED_PIPELINES.register_module()
class LoadEmptyAnnotations:
    def __init__(self, with_bbox=False, with_mask=False, with_seg=False, fill_value=255):
        """Load Empty Annotations for un-supervised pipeline"""
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.fill_value = fill_value

    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 5))
            results["gt_labels"] = np.zeros((0,))
            results["gt_bboxes_conformal"] = np.zeros((0, 5))
            results["gt_labels_conformal"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            raise NotImplementedError
        if self.with_seg:
            results["gt_semantic_seg"] = self.fill_value * np.ones(
                results["img"].shape[:2], dtype=np.uint8)
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results

@ROTATED_PIPELINES.register_module()
class LoadConformalAnnotations:

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_conformal = ann_info.get('bboxes_conformal', None)
        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        if gt_bboxes_conformal is not None:
            results['gt_bboxes_conformal'] = gt_bboxes_conformal.copy()
            results['bbox_fields'].append('gt_bboxes_conformal')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):

        results['gt_labels'] = results['ann_info']['labels'].copy()
        results['gt_labels_conformal'] = results['ann_info']['labels_conformal'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
    
@ROTATED_PIPELINES.register_module()
class DefaultFormatBundleConformal:

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            results['img'] = DC(
                img, padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_conformal', 'gt_labels_conformal']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'

@ROTATED_PIPELINES.register_module()
class EmptyPolyRandomRotate(PolyRandomRotate):

    def __call__(self, results):
        """Call function of EmptyPolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)

        return results


@ROTATED_PIPELINES.register_module()
class ExtraAttrs:
    def __init__(self, **attrs):
        self.keep_raw = attrs.pop('keep_raw', False)
        self.attrs = attrs

    def __call__(self, results):
        if self.keep_raw:
            results['raw_img'] = results['img'].copy()
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


class DTGaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0]):
        self.rad_range = rad_range

    def __call__(self, x):
        rad = np.random.uniform(*self.rad_range)
        x = x.filter(ImageFilter.GaussianBlur(radius=rad))
        return x


DT_LOCAL_AUGS = {
    'DTGaussianBlur': DTGaussianBlur
}


def build_dt_aug(type, **kwargs):
    return DT_LOCAL_AUGS[type](**kwargs)
