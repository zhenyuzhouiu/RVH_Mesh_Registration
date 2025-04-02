"""
base smpl fitter class to handle data io, load smpl, output saving etc. so that they can be easily reused later
this can be inherited for fitting smplh, smph+d to scan, kinect point clouds etc.

Author: Xianghui, 12, January 2022
"""
import torch
from os.path import join, split, splitext
from pytorch3d.structures import Meshes
from pytorch3d.io import save_ply, load_ply, load_obj
import pickle as pkl
import numpy as np
import json
from psbody.mesh import MeshViewer, Mesh
from lib.smpl.priors.th_hand_prior import mean_hand_pose
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.const import *


class BaseFitter(object):
    def __init__(self, model_root, device='cuda:0', save_name='smpl', debug=False, hands=True):
        self.model_root = model_root # root path to the smpl or smplh model
        self.debug = debug
        self.save_name = save_name # suffix of the output file
        self.device = device
        self.hands = hands
        self.save_name_base = 'smplh' if hands else 'smpl'
        if debug:
            self.mv = MeshViewer(window_width=512, window_height=512)
        if self.hands:
            print("Using SMPL-H model for registration")
        else:
            print("Using SMPL model for registration")

    def fit(self, scans, pose_files, gender='male', save_path=None):
        raise NotImplemented

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        """
        optimize smpl pose and shape parameters together
        Args:
            th_scan_meshes:
            smpl:
            iterations:
            steps_per_iter:
            th_pose_3d:

        Returns:

        """
        raise NotImplemented

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        """
        Initially we want to only optimize the global rotation of SMPL. Next we optimize full pose.
        We optimize pose based on the 3D keypoints in th_pose_3d.
        Args:
            th_scan_meshes:
            smpl:
            iterations:
            steps_per_iter:
            th_pose_3d:
            prior_weight:

        Returns:

        """
        raise NotImplemented

    def init_smpl(self, batch_sz, gender, pose=None, betas=None, trans=None, flip=False):
        """Initialize a batch of smpl model

        Args:
            batch_sz (_type_): _description_
            gender (_type_): _description_
            pose (_type_, optional): pose parameters for pose blened shape. Defaults to None.
            betas (_type_, optional): shape parameters for shape blended shape. Defaults to None.
            trans (_type_, optional): _description_. Defaults to None.
            flip (bool, optional): rotate smpl around z-axis by 180 degree, required for kinect point clouds, which has different coordinate from scans. Defaults to False.

        Returns:
            _type_: batch smplh model
        """
        # ===== Model parameter betas, pose, and trans initialization  =====
        # region
        # sp = SmplPaths(gender=gender)
        # smpl_faces = sp.get_faces()
        # th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(self.device)
        num_betas = 10  # smplh only allows 10 shape parameters
        # get the prior mean and covariance of the pose parameters
        # and it will use the Mahalanobis distance to measure the difference between the pose parameters and the prior
        prior = get_prior(self.model_root, gender=gender, device=self.device) 
        total_pose_num = SMPLH_POSE_PRAMS_NUM if self.hands else SMPL_POSE_PRAMS_NUM
        pose_init = torch.zeros((batch_sz, total_pose_num))
        if pose is None:
            # initialize hand pose from mean TODO by Zhenyu: (1 head + 21 body + 2 hand = 24 keypoints)?
            pose_init[:, 3:SMPLH_HANDPOSE_START] = prior.mean
            if self.hands:
                # from the priors get the left hand and right hand mean pose
                hand_mean = mean_hand_pose(self.model_root) 
                hand_init = torch.tensor(hand_mean, dtype=torch.float).to(self.device)
            else:
                hand_init = torch.zeros((batch_sz, SMPL_HAND_POSE_NUM))
            pose_init[:, SMPLH_HANDPOSE_START:] = hand_init
            if flip:
                pose_init[:, 2] = np.pi
        else:
            pose_init[:, :SMPLH_HANDPOSE_START] = pose[:, :SMPLH_HANDPOSE_START]
            if pose.shape[1] == total_pose_num:
                pose_init[:, SMPLH_HANDPOSE_START:] = pose[:, SMPLH_HANDPOSE_START:]
        beta_init = torch.zeros((batch_sz, num_betas)) if betas is None else betas
        trans_init = torch.zeros((batch_sz, 3)) if trans is None else trans
        betas, pose, trans = beta_init, pose_init, trans_init
        # endregion
        
        # Init SMPL, pose with mean smpl pose, as in ch.registration
        # smpl = SMPLHPyTorchWrapperBatch(self.model_root, batch_sz, betas, pose, trans,
        #                                 num_betas=num_betas, device=self.device, gender=gender).to(self.device)
        smpl = SMPLPyTorchWrapperBatch(self.model_root, 
                                       batch_sz,
                                       betas, 
                                       pose, 
                                       trans,
                                       num_betas=num_betas, 
                                       device=self.device,
                                       gender=gender, 
                                       hands=self.hands).to(self.device)
        return smpl

    @staticmethod
    def load_smpl_params(pkl_files):
        """
        load smpl params from file
        Args:
            pkl_files:

        Returns:

        """
        pose, betas, trans = [], [], []
        for spkl in pkl_files:
            smpl_dict = pkl.load(open(spkl, 'rb'), encoding='latin-1')
            p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
            pose.append(p) # smplh only allows 10 shape parameters
            # if len(b) == 10:
            #     temp = np.zeros((300,))
            #     temp[:10] = b
            #     b = temp.astype('float32')
            betas.append(b)
            trans.append(t)
        pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)
        return pose, betas, trans

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

    def save_outputs(self, save_path, scan_paths, smpl, th_scan_meshes, save_name='smpl'):
        th_smpl_meshes = self.smpl2meshes(smpl)
        mesh_paths, names = self.get_mesh_paths(save_name, save_path, scan_paths)
        self.save_meshes(th_smpl_meshes, mesh_paths)
        # self.save_meshes(th_scan_meshes, [join(save_path, n) for n in names]) # save original scans
        # Save params
        self.save_smpl_params(names, save_path, smpl, save_name)
        return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy()

    def smpl2meshes(self, smpl):
        "convert smpl batch to pytorch3d meshes"
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        return th_smpl_meshes

    def get_mesh_paths(self, save_name, save_path, scan_paths):
        names = [split(s)[1] for s in scan_paths]
        # Save meshes
        mesh_paths = []
        for n in names:
            if n.endswith('.obj'):
                mesh_paths.append(join(save_path, n.replace('.obj', f'_{save_name}.ply')))
            else:
                mesh_paths.append(join(save_path, n.replace('.ply', f'_{save_name}.ply')))
        return mesh_paths, names

    def save_smpl_params(self, names, save_path, smpl, save_name):
        for p, b, t, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                              smpl.trans.cpu().detach().numpy(), names):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t}
            sfx = splitext(n)[1]
            pkl_file = join(save_path, n.replace(sfx, f'_{save_name}.pkl'))
            pkl.dump(smpl_dict, open(pkl_file, 'wb'))
            print('SMPL parameters saved to', pkl_file)

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def save_meshes(meshes, save_paths):
        print('Mesh saved at', save_paths[0])
        for m, s in zip(meshes, save_paths):
            save_ply(s, m.verts_list()[0].cpu(), m.faces_list()[0].cpu())

    def load_j3d(self, pose_files):
        """
        load 3d body, face, and hand keypoints with total 137 different keypoints
        Args:
            pose_files: json files containing the body keypoints location

        Returns: a list of body keypoints

        """
        joints = []
        for file in pose_files:
            data = json.load(open(file))
            J3d = np.array(data).reshape((-1, 4))
            joints.append(J3d)
        joints = np.stack(joints)
        return torch.from_numpy(joints).float().to(self.device)

    @staticmethod
    def load_scans(scans, device='cuda:0', ret_cent=False):
        """Load the scans as pytorch3d meshes, and return the centers of the scans if required.

        Args:
            scans (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda:0'.
            ret_cent (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        verts, faces, centers = [], [], []
        for scan in scans:
            print('scan path ...', scan)
            if scan.endswith('.ply'):
                v, f = load_ply(scan)
            else:
                v, f, _ = load_obj(scan)
                f = f[0]  # see pytorch3d doc
            verts.append(v)
            faces.append(f)
            centers.append(torch.mean(v, 0))
        th_scan_meshes = Meshes(verts, faces).to(device) # pytorch3d meshes
        if ret_cent:
            return th_scan_meshes, torch.stack(centers, 0).to(device)
        return th_scan_meshes

    def viz_fitting(self, smpl, th_scan_meshes, ind=0,
                    smpl_vc=np.array([0, 1, 0]), **kwargs):
        verts, _, _, _ = smpl()
        smpl_mesh = Mesh(v=verts[ind].cpu().detach().numpy(), 
                         f=smpl.faces.cpu().numpy())
        scan_mesh = Mesh(v=th_scan_meshes.verts_list()[ind].cpu().detach().numpy(),
                         f=th_scan_meshes.faces_list()[ind].cpu().numpy(), vc=smpl_vc)
        self.mv.set_dynamic_meshes([scan_mesh, smpl_mesh])

    def copy_smpl_params(self, split_smpl, smpl):
        smpl.pose.data[:, :3] = split_smpl.global_pose.data
        smpl.pose.data[:, 3:66] = split_smpl.body_pose.data
        smpl.pose.data[:, 66:] = split_smpl.hand_pose.data
        smpl.betas.data[:, :2] = split_smpl.top_betas.data
        smpl.betas.data[:, 2:] = split_smpl.other_betas.data

        smpl.trans.data = split_smpl.trans.data
