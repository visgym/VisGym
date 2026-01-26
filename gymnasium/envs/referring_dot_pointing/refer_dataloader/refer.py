__author__ = 'licheng'

"""
This interface provides access to five datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
5) sunspot
split by unc, google, and boulder

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
getLandIds - get landmark ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
loadLand   - load landmarks with specified landmark ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from pycocotools import mask
from collections import defaultdict

class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(self.DATA_DIR, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        elif dataset == 'sunspot':
            self.IMAGE_DIR = osp.join(data_root, 'images/')
        elif dataset == 'syntheticsun':
            self.IMAGE_DIR = osp.join(data_root, 'images/syntheticsun')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        with open(ref_file, 'rb') as f:
            print("Loading %s" % ref_file)
            self.data['refs'] = pickle.load(f)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        landmarks_file = osp.join(self.DATA_DIR, 'landmarks.json')
        if osp.exists(landmarks_file):
            landmarks = json.load(open(landmarks_file, 'r'))
            self.data['landmarks'] = landmarks

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, Land, imgToAnns = {}, {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']
        if 'landmarks' in self.data:
            for sent_id, land in self.data['landmarks'].items():
                for phrase_id, phrase in land['phrases'].items():
                    Land[phrase_id] = phrase
                    Land[phrase_id]['sent_id'] = sent_id
                    Land[phrase_id]['image_id'] = land['image_id']


        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}

        # Remove any referring expressions that don't have a corresponding annotation
        self.data['refs'] = [ref for ref in self.data['refs'] if ref['ann_id'] in Anns]
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        landToRefs, refToLand, landToSent, landToAnn = {}, {}, {}, {}
        sentToLand = defaultdict(list)
        for land_id, land in Land.items():
            # landToRefs[land_id] = sentToRef[land['sent_id']]
            # landToSent[land_id] = land['sent_id']
            landToAnn[land_id] = land['phrase_id']
            # refToLand[sentToRef[land['sent_id']]['ref_id']] = land_id
            sentToLand[land['sent_id']].append(land_id)

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.Land = Land
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        self.sentToLand = sentToLand
        self.landToAnn = landToAnn
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [ref for image_id in image_ids for ref in self.imgToRefs[image_id]]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ann_ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getSentIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            sent_ids = list(set([id for ref_id in ref_ids for id in self.Refs[ref_id]['sent_ids']]))
        else:
            sent_ids = self.Imgs.keys()
        return sent_ids

    def getLandIds(self, sent_ids=[]):
        sent_ids = sent_ids if type(sent_ids) == list else [sent_ids]
        if len(sent_ids) == 0:
            land_ids = [land['id'] for land in self.Land]
        else:
            land_ids = []
            for sent_id in sent_ids:
                land_ids.extend(self.sentToLand[sent_id])
        return land_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        else:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        else:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        else:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        else:
            return [self.Cats[cat_ids]]

    def loadLand(self, land_ids=[]):
        if type(land_ids) == list:
            return [self.Land[land_id] for land_id in land_ids]
        else:
            return [self.Land[land_ids]]

    def showRef(self, ref, seg_box='seg'):
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        image = self.Imgs[ref['image_id']]
        self.plotAnnotationsOnImg(image, [ref['ann_id']], seg_box)

    def getMask(self, ref=[], ann=[]):
        # return mask, area and mask-center
        if len(ref)>0 and len(ann)==0:
            ann = self.refToAnn[ref['ref_id']]
            image = self.Imgs[ref['image_id']]
        elif len(ann)>0 and len(ref)==0:
            image = self.Imgs[ann['image_id']]
        else:
            raise ValueError('Only one of ref or ann arguments should be assigned when calling getMask')

        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']
        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)

    def showLandmarkMasks(self, sent):
        lands = self.sentToLand(sent['sent_id'])
        msk = None
        for land in lands:
            M = self.getMask(ann=self.Anns[land['phrase_id']])
            if msk is None:
                msk = M['mask']
            else:
                msk = msk | M['mask']
        ax = plt.gca()
        ax.imshow(msk)

    def showLandmarks(self, sent, seg_box='seg'):
        # show refer expression
        print('%s: %s' % (sent['sent_id'], sent['sent']))
        # show landmarks
        land_ids = self.getLandIds(sent_ids=[sent['sent_id']])

        print('Landmarks')
        lands = self.loadLand(land_ids)
        for i, land in enumerate(lands):
            print("{}. {}".format(i+1, sent['tokens'][land['token_offsets'][0][0]:land['token_offsets'][0][1] + 1]))

        ref = self.sentToRef[sent['sent_id']]
        image = self.Imgs[ref['image_id']]
        ann_ids = [self.landToAnn[land_id] for land_id in land_ids]
        self.plotAnnotationsOnImg(image, ann_ids, seg_box)

    def plotAnnotationsOnImg(self, image, ann_ids, seg_box='seg'):
        ax = plt.gca()
        # show image
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)

        if seg_box == 'seg':
            for ann_id in ann_ids:
                try:
                    ann = self.Anns[ann_id]
                    polygons = []
                    color = []
                    c = 'none'
                    if type(ann['segmentation']) == list:
                        # polygon used for refcoco*
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((len(seg) // 2, 2))
                            polygons.append(Polygon(poly,  closed=True, alpha=0.4))
                            color.append(c)
                        p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                        ax.add_collection(p)  # thick yellow polygon
                        p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                        ax.add_collection(p)  # thin red polygon
                    else:
                        # mask used for refclef
                        rle = ann['segmentation']
                        m = mask.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                except KeyError as e:
                    print('Missing segmentation: {}'.format(ann_id))
        # show bounding-box
        elif seg_box == 'box':
            for ann_id in ann_ids:
                ann = self.Anns[ann_id]
                bbox = ann['bbox']
                box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
                ax.add_patch(box_plot)