"""Raw metre metrics and combined-overlay diagnostic boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import evaluate_parking_lidar_depth as evaluation


class ParkingLidarDepthEvaluationTests(unittest.TestCase):
    def test_raw_metres_are_not_scale_aligned(self):
        result=evaluation.metrics([2,4,6],[4,8,12])
        self.assertEqual(result['mean_absolute_error_m'],4)
        self.assertEqual(result['mean_signed_error_m'],4)
        self.assertEqual(result['abs_rel'],1)
        self.assertEqual(result['fraction_within_0_10m'],0)
        self.assertEqual(result['delta1'],0)

    def test_invalid_and_empty_metrics(self):
        self.assertIsNone(evaluation.metrics([],[]))
        for ref,pred in [([2],[0]),([2],[float('nan')]),([0],[2]),([2,3],[2])]:
            with self.assertRaises(ValueError):evaluation.metrics(ref,pred)
        self.assertEqual(evaluation.metrics([2,3],[2,3])['fraction_within_0_10m'],1)

    def test_same_gt_only_class_range_and_static_masks(self):
        p=dict(z=np.array([2.,3.,4.,6.,6.1]),semantic=np.array([17,2,17,2,17]),
            conservative=np.array([True,True,False,True,True]),static=np.array([True,False,True,True,True]))
        mapping={'Vehicle':[17],'Pedestrian':[2]}
        np.testing.assert_array_equal(evaluation.mask(p,mapping,'conservative','Pedestrian',2,6,True),[False,True,False,True,False])
        np.testing.assert_array_equal(evaluation.mask(p,mapping,'conservative_static_objects','Pedestrian',2,6,True),[False,False,False,True,False])
        self.assertEqual(evaluation.mask(p,mapping,'conservative','All',2,6).sum(),2)

    def test_point_coverage_cannot_cross_class_or_invent_depth(self):
        p=dict(z=np.array([2.,4.,6.]),xy=np.array([[10,10],[20,20],[30,30]]))
        rows=[dict(coco_class_id=0,bbox=[5,5,25,25]),dict(coco_class_id=2,bbox=[25,25,35,35])]
        result=evaluation.urban_coverage(p,rows)
        np.testing.assert_array_equal(result['Pedestrian'],[True,True,False])
        np.testing.assert_array_equal(result['Vehicle'],[False,False,True])
        self.assertNotIn('depth',result)

    def test_summary_exposes_empty_cells_and_image_weighting(self):
        config=dict(range_bands_m=[[2,3],[3,4],[4,5],[5,6],[6,10],[10,20],[20,40],[40,80]],
            minimum_points_for_descriptive_cell=100,minimum_images_for_descriptive_cell=5)
        manifest=dict(config=config,mapping={'Vehicle':[17],'Pedestrian':[2]})
        data=[]
        for n,pred in [(10,3.),(1,5.)]:
            p=dict(z=np.full(n,2.),semantic=np.full(n,17),conservative=np.ones(n,bool),static=np.ones(n,bool))
            predictions={k:np.full(n,pred) for k in evaluation.MODELS}
            data.append((dict(scene='s'),p,predictions))
        result=evaluation.summarize(manifest,data)['conservative']
        self.assertIsNone(result['Pedestrian']['2-6']['models'][evaluation.MODELS[0]])
        cell=result['Vehicle']['2-6']
        self.assertEqual(cell['points'],11);self.assertEqual(cell['images'],2)
        self.assertFalse(cell['descriptive_coverage_adequate'])
        score=cell['models'][evaluation.MODELS[0]]
        self.assertAlmostEqual(score['mean_absolute_error_m'],13/11)
        self.assertEqual(score['macro_image_mean_absolute_error_m'],2)

    def test_frozen_input_hashes_reject_changed_gt_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);root=base/'dataset';folder=base/'prepared'
            root.mkdir();folder.mkdir()
            config=dict(maximum_images=1)
            config_file=base/'config.json'
            config_file.write_bytes(evaluation.common.encoded(config))
            for name in ['image','cloud','labels']:(root/name).write_bytes(name.encode())
            (folder/'gt.npz').write_bytes(b'frozen sparse points')
            sample=dict(sample_token='sample',camera_token='camera',ground_truth_file='gt.npz',
                ground_truth_sha256=evaluation.common.file_sha(folder/'gt.npz'))
            for name in ['image','cloud','labels']:
                sample[name+'_file']=name;sample[name+'_sha256']=evaluation.common.file_sha(root/name)
            identity=dict(mapping={'Vehicle':[17]},config_sha256=evaluation.common.file_sha(config_file),table_sha256={})
            (folder/'prepare_identity.json').write_bytes(evaluation.common.encoded(identity))
            manifest=dict(complete=True,dataset_root=str(root),config=config,mapping=identity['mapping'],
                prepare_identity_sha256=evaluation.common.identity(identity),samples=[sample])
            manifest['manifest_sha256']=evaluation.common.identity(manifest)
            saved=evaluation.common.encoded(manifest)
            (folder/'input_manifest.json').write_bytes(saved)
            with patch.object(evaluation.ground_truth,'CONFIG',config_file):
                self.assertEqual(evaluation.read_manifest(folder),manifest)
                (folder/'gt.npz').write_bytes(b'changed points')
                with self.assertRaisesRegex(ValueError,'GT cache changed'):evaluation.read_manifest(folder)
                manifest['config']['maximum_images']=2
                (folder/'input_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'manifest changed'):evaluation.read_manifest(folder)


if __name__=='__main__':unittest.main()
