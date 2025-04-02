"""
fit smplh to scans

crated by Xianghui, 12, January 2022

the code is tested
"""

import sys, os

import torch.backends
sys.path.append(os.getcwd())
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from smpl_registration.base_fitter import BaseFitter
from lib.body_objectives import batch_get_pose_obj, batch_3djoints_loss
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams



class SMPLHFitter(BaseFitter):
    def __init__(self, model_root, device='cuda:0', save_name='smpl', debug=False, hands=True):
        super().__init__(model_root, device, save_name, debug, hands)
        
    def fit(self, scans, pose_files, gender='male', save_path=None):
        """First optimize the pose only, then optimize the pose and shape

        Args:
            scans (_type_): _description_
            pose_files (_type_): _description_
            gender (str, optional): _description_. Defaults to 'male'.
            save_path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Batch size
        batch_sz = len(scans)

        # Load scans and return the center of it, but not centerized the scaned mesh. 
        # Once smpl is registered, move it accordingly.
        th_scan_meshes, centers = self.load_scans(scans, device=self.device, ret_cent=True)

        # Initialization smpl by the center of the scaned mesh without the betas and pose 
        smpl = self.init_smpl(batch_sz, gender, trans=centers)

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 5, 5, 30, 30

        # Load 3D pose from detected 25+70+42 3D joints with confidence score
        th_pose_3d = None
        if pose_files is not None:
            th_pose_3d = self.load_j3d(pose_files)

            # Optimize pose first
            self.optimize_pose_only(th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, th_pose_3d)

        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        # Optimizer
        # for split_smpl, it will optimize the split_smpl.trans, split_smpl.top_betas, and split_smpl.global_pose
        optimizer = torch.optim.Adam([smpl.trans, 
                                      smpl.betas, 
                                      smpl.pose], 
                                     0.02, 
                                     betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')
    
    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        """The first iter_for_global iterations only update the split_smpl.trans, split_smpl.top_betas, and split_smpl.global_pose.
        And the rest iterations will update the split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose, and split_smpl.body_pose.

        Args:
            th_scan_meshes (_type_): _description_
            smpl (_type_): _description_
            iterations (_type_): _description_
            steps_per_iter (_type_): _description_
            th_pose_3d (_type_): _description_
            prior_weight (_type_, optional): _description_. Defaults to None.
        """
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, 
                                      split_smpl.top_betas, 
                                      split_smpl.global_pose],
                                     0.02, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # for the first iter_for_global iterations, optimize global orientation only, 
        # then optimize full pose
        iter_for_global = 5  # TODO by Zhenyu: Why do the first 5 iterations only update the global pose?
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation with trans, top_betas, and global_pose
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose with trans, top_betas, global_pose, and body_pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                optimizer = torch.optim.Adam([split_smpl.trans, 
                                              split_smpl.top_betas, 
                                              split_smpl.global_pose,
                                              split_smpl.body_pose],
                                             0.02, 
                                             betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                # it will calculate the pose_pr, betas, and pose_obj (MSE of joint location) losses
                loss_dict = self.forward_step_pose_only(split_smpl,
                                                        th_pose_3d,
                                                        prior_weight)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it/2)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)

        print('** Optimised smpl pose **')

    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender, device=self.device)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, 
                                faces=torch.stack([smpl.faces] * len(verts), 
                                                  dim=0))

        # losses
        loss = dict()
        # point to the cloest face distance + face to the cloest point distance
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, 
                                               Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, 
                                               Pointclouds(points=th_smpl_meshes.verts_list()))
        loss['betas'] = torch.mean(smpl.betas ** 2)  # reguralization term
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
        if th_pose_3d is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss['pose_obj'] = j3d_loss
            # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        return loss
    
    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """Performs a forward step, given smpl and scan meshes. Then computes the losses mainly based on the pose_pr, betas, and pose_obj losses without any forward step of SMPL_Layer.
        currently no prior weight implemented for smplh

        Args:
            smpl (_type_): _description_
            th_pose_3d (_type_): the pre-detected 3D joints of body, face, and hands
            prior_weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender, device=self.device)

        # losses
        loss = dict()
        # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        # 3D joints loss
        J, face, hands = smpl.get_landmarks() # get 3D joints from smpl under the current pose 
        joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        # using the Mahalanobis distance to measure the difference between pose and prior
        loss['pose_pr'] = torch.mean(prior(smpl.pose)) 
        # betas for shape, reguralization term
        loss['betas'] = torch.mean(smpl.betas ** 2) 
        j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
        loss['pose_obj'] = j3d_loss
        return loss

    def compose_smpl_joints(self, J, face, hands, th_pose_3d):
        """Combined the predicted 3D joints from SMPL by the shape of th_pose_3d. If th_pose_3d is 25, then only body joints are used. If th_pose_3d is 25+70+42, then body, face, and hands joints are used.

        Args:
            J (_type_): _description_
            face (_type_): _description_
            hands (_type_): _description_
            th_pose_3d (_type_): _description_

        Returns:
            _type_: _description_
        """
        if th_pose_3d.shape[1] == 25:
            joints = J
        else:
            joints = torch.cat([J, face, hands], 1)
        return joints

    def copy_smpl_params(self, smpl, split_smpl):
        """Put back pose, shape and trans of split_smpl into original smpl

        Args:
            smpl (_type_): source smpl
            split_smpl (_type_): _description_
        """
        
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def get_loss_weights(self):
        """Set loss weights"""
        # s2m and m2s are the distance (point to face + face to point distance) between scan and registered smpl model
        loss_weight = {'s2m': lambda cst, it: 20. ** 2 * cst * (1 + it), # scan to smpl model
                       'm2s': lambda cst, it: 20. ** 2 * cst / (1 + it), 
                       'betas': lambda cst, it: 10. ** 1.0 * cst / (1 + it), # for shape
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it), # for prior pose
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it) # for 3D joints loss
                       }
        return loss_weight


def main(args):
    if sys.platform.startswith("darwin"):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    fitter = SMPLHFitter(args.model_root, device=device, debug=args.display, hands=args.hands)
    fitter.fit([args.scan_path], [args.pose_file], args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--scan_path', type=str, help='path to the 3d scans', default='data/mesh_1/scan.obj')
    parser.add_argument('--pose_file', type=str, help='detected 3d body joints file by openpose', default="data/mesh_1/3D_pose.json")
    parser.add_argument('--save_path', type=str, help='save path for all scans', default='data/mesh_1')
    parser.add_argument('--gender', type=str, default='male_v.1.0.0')  # can be female
    parser.add_argument('--display', default=True, action='store_true')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument('--hands', default=False, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)