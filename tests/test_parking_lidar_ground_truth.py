"""Projection, occlusion ordering and GT-only sampling guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts import prepare_parking_lidar_depth as gt


def config():return json.loads(gt.CONFIG.read_text())


class LidarGroundTruthTests(unittest.TestCase):
    def test_pinhole_projection_and_pose_translation(self):
        identity=dict(rotation=[1,0,0,0],translation=[0,0,0])
        camera=dict(identity,camera_intrinsic=[[100,0,100],[0,100,50],[0,0,1]])
        xyz=np.array([[0,0,5],[1,0,5],[0,0,-5]],dtype=np.float32)
        uv,z,ids,_=gt.camera_projection(xyz,identity,identity,identity,camera,200,100)
        np.testing.assert_allclose(uv,[[100,50],[120,50]])
        np.testing.assert_array_equal(ids,[0,1]);np.testing.assert_allclose(z,[5,5])
        moved=dict(identity,translation=[1,0,0])
        uv,_,_,_=gt.camera_projection(xyz,identity,identity,moved,camera,200,100)
        np.testing.assert_allclose(uv,[[80,50],[100,50]])

    def test_wxyz_rotation_convention(self):
        q=[2**-.5,0,0,2**-.5]
        np.testing.assert_allclose(gt.rotation(q)@np.array([1,0,0]),[0,1,0],atol=1e-8)
        for q in [[0,0,0,0],[1,0,0],[float('nan'),0,0,1]]:
            with self.assertRaises(ValueError):gt.rotation(q)

    def test_nearest_return_wins_before_noise_filter(self):
        c=config();uv=np.array([[10.,10.],[10.,10.],[20.,20.]])
        p=gt.rasterize(uv,np.array([3.,5.,4.]),np.array([0,17,17]),np.zeros(3,dtype=bool),50,50,c)
        self.assertEqual(len(p['z']),1)
        self.assertEqual(p['z'][0],4)
        np.testing.assert_array_equal(p['xy'],[[20,20]])

    def test_duplicate_depth_order_and_gt_only_boundary_guard(self):
        c=config();uv=np.array([[10.,10.],[10.,10.],[11.,10.],[30.,30.]])
        p=gt.rasterize(uv,np.array([5.,3.,7.,4.]),np.full(4,17),np.ones(4,dtype=bool),50,50,c)
        np.testing.assert_allclose(p['z'],[3,7,4])
        np.testing.assert_array_equal(p['conservative'],[False,False,True])
        self.assertTrue(p['static'].all())

    def test_selection_uses_gt_not_predictions_and_keeps_keyframe_groups(self):
        c=config();c['maximum_images']=3
        def row(token,sample,points):
            counts={f'{cls}:{low}-{high}':0 for cls in ['Vehicle','Pedestrian'] for low,high in c['range_bands_m']}
            counts['Vehicle:2-3']=points
            return dict(camera_token=token,sample_token=sample,counts=counts)
        rows=[row('a','one',10),row('b','one',20),row('c','two',10),row('d','three',0)]
        selected=gt.select(rows,c)
        self.assertEqual(len(selected),3)
        self.assertEqual(len({r['sample_token'] for r in selected}),3)
        self.assertEqual([r['camera_token'] for r in gt.select(rows[::-1],c)], [r['camera_token'] for r in selected])

    def test_range_and_semantic_coverage_do_not_invent_missing_points(self):
        c=config();points=dict(z=np.array([2.,3.,6.,6.1]),semantic=np.array([17,2,2,17]),conservative=np.ones(4,dtype=bool))
        counts=gt.cell_counts(points,{'Vehicle':[17],'Pedestrian':[2]},c)
        self.assertEqual(counts['Vehicle:2-6'],1);self.assertEqual(counts['Pedestrian:2-6'],2)
        self.assertEqual(counts['Pedestrian:2-3'],0)

    def test_dataset_path_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'sample').write_bytes(b'abc')
            self.assertEqual(gt.safe_file(root,'sample'),root/'sample')
            for name in ['../escape','/tmp/escape','a\\b']:
                with self.assertRaises(ValueError):gt.safe_file(root,name)


if __name__=='__main__':unittest.main()
