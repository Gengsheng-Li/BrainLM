import os.path

import numpy as np

import nibabel as nib
import scipy.io as scio
from nilearn import surface

import pygeodesic.geodesic as geodesic
from scipy.sparse import csr_matrix, load_npz, save_npz


def make_topodis():
    all_vert_num = 32492

    label_A424 = nib.load('datasets/A424.dlabel.nii')
    label_A424 = label_A424.get_fdata().astype(int).squeeze()
    print(label_A424.shape)  # (91k)

    tmp = scio.loadmat('surf/cifti_vert_LR.mat')
    for hemi, start_ind in [['L', 181],
                 ['R', 1]]:
    # for hemi, start_ind in [['R', 1],]:

        start, count, vertlist = \
            tmp['start_'+hemi].squeeze(), tmp['count_'+hemi].squeeze(), tmp['vertlist_'+hemi].squeeze(),
        print(start, count, vertlist)

        vertices, faces = surface.load_surf_data('surf/fsaverage.{}.midthickness.32k_fs_LR.surf.gii'.format(hemi))
        print(vertices.shape, faces.shape)
        geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

        # Create a sparse adjacency matrix
        if not os.path.exists("surf/adjacency_matrix_{}.npz"):
            N = len(vertices)
            adjacency_matrix = np.zeros((N,N), dtype=bool)
            for face in faces:
                # Set the entries of the adjacency matrix based on the distances between connected vertices
                for i in range(3):
                    for j in range(i + 1, 3):
                        vertex1 = face[i]
                        vertex2 = face[j]
                        adjacency_matrix[vertex1, vertex2] = True
                        adjacency_matrix[vertex2, vertex1] = True
            adjacency_matrix = csr_matrix(adjacency_matrix)
            save_npz("surf/adjacency_matrix_{}.npz".format(hemi), adjacency_matrix)
        else:
            adjacency_matrix = load_npz("surf/adjacency_matrix_{}.npz".format(hemi))

        label_32k = np.zeros(all_vert_num, dtype=int)
        label_32k[vertlist] = label_A424[start-1: start + count-1]

        dists = []
        for ind in range(start_ind, start_ind+180):
        # for ind in range(start_ind, start_ind+2):
            ind_vert = np.where(label_32k == ind)[0]
            # print(ind_vert)
            adj = adjacency_matrix[ind_vert,:]
            # print(adj.shape)

            border = []
            for i in range(adj.shape[0]):
                # print(adj[i,:].nonzero()[1])
                vii = label_32k[adj[i,:].nonzero()[1]] != ind  # (1,1,0,0, 0) 1 - is border
                # print(vii)
                if vii.any():
                    border.append(i)

            border = np.array(border, dtype=int)
            # print(border)
            # print(ind_vert[border])

            dist_map, _ = geoalg.geodesicDistances(ind_vert[border], np.arange(all_vert_num, dtype=int))
            print(dist_map.shape)
            dist_map[ind_vert] = - dist_map[ind_vert]
            dist_map /= np.abs(dist_map.min())
            print(dist_map.shape, dist_map.min(), dist_map.max())

            dists.append(dist_map)

        dists = np.stack(dists, axis=1)  # (V, K)
        print(dists.shape)
        np.save("surf/distance_map.{}.npy".format(hemi), dists[vertlist,:])

        template = nib.load("surf/fsaverage.{}.Glasser.32k_fs_LR.label.gii".format(hemi))
        template.remove_gifti_data_array(0)
        template.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(dists.astype(np.float32)))
        nib.loadsave.save(template, 'surf/distance_map.{}.32k_fs_LR.func.gii'.format(hemi))


import torch
import torch.nn as nn


def scipycoo2torchcoo(sp):
    vals = sp.data
    indices = np.vstack((sp.row, sp.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vals)
    shape = sp.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class refineBLM_loss(nn.Module):
    def __init__(self, lam_atlas, lam_smooth, device='cuda'):
        super(refineBLM_loss, self).__init__()
        self.lam_atlas = lam_atlas
        self.lam_smooth = lam_smooth
        self.mse = nn.MSELoss()

        dist_L = np.load('surf/distance_map.L.npy')
        dist_R = np.load('surf/distance_map.R.npy')
        self.dist_L = torch.from_numpy(dist_L).to(device)  # (V_L, 180)
        self.dist_R = torch.from_numpy(dist_R).to(device)  # (V_R, 180)

        tmp = scio.loadmat('surf/cifti_vert_LR.mat')
        vertlist_L, vertlist_R = tmp['vertlist_L'].squeeze(), tmp['vertlist_R'].squeeze(),

        adjacency_L = load_npz("surf/adjacency_matrix_L.npz")[vertlist_L, :][:, vertlist_L].tocoo()
        adjacency_R = load_npz("surf/adjacency_matrix_R.npz")[vertlist_R, :][:, vertlist_R].tocoo()
        self.adj_L = scipycoo2torchcoo(adjacency_L).to_sparse_csr().to(device)  # (V_L, V_L)
        self.adj_R = scipycoo2torchcoo(adjacency_R).to_sparse_csr().to(device)  # (V_L, V_L)
        print(self.adj_R.shape)

    def loss_predict(self, pred, targ):
        """
        :param pred: torch cuda (B, T, 424), predicted
        :param targ: torch cuda (B, T, 424)
        """
        return self.mse(pred, targ)

    def loss_atlas_simi(self, assign_L, assign_R):
        """
        :param assign_L: torch cuda (V_L, 180), softmaxed, range 0-1
        :param assign_R: torch cuda (V_R, 180), softmaxed, range 0-1
        """
        loss = 0
        loss += (assign_L * self.dist_L).sum() / assign_L.shape[0]
        loss += (assign_R * self.dist_R).sum() / assign_R.shape[0]
        return loss / 2

    def loss_smooth(self, assign_L, assign_R):
        """
        :param assign_L: torch cuda (V_L, 180), softmaxed, range 0-1
        :param assign_R: torch cuda (V_R, 180), softmaxed, range 0-1
        """
        loss = 0
        loss += self.mse(assign_L, torch.sparse.mm(self.adj_L, assign_L))
        loss += self.mse(assign_R, torch.sparse.mm(self.adj_R, assign_R))
        return loss / 2

    # def forward(self, pred, targ, assign_L, assign_R):
    #     loss_pred = self.loss_predict(pred, targ)
    #     loss_atlas = self.loss_atlas_simi(assign_L, assign_R) if self.lam_atlas > 0 else 0
    #     return loss_pred + self.lam_atlas * loss_atlas, loss_pred, loss_atlas

    def forward(self, pred, targ, assign_L, assign_R):
        loss_pred = self.loss_predict(pred, targ)
        loss_atlas = self.loss_atlas_simi(assign_L, assign_R) if self.lam_atlas > 0 else 0
        loss_smooth = self.loss_smooth(assign_L, assign_R) if self.lam_smooth > 0 else 0
        return (loss_pred + self.lam_atlas * loss_atlas + self.lam_smooth * loss_smooth,
                loss_pred, loss_atlas, loss_smooth)


if __name__ == '__main__':
    # make_topodis()
    device = 'cuda'
    B, VL, VR = 8, 29696, 29716
    Ka = 424
    Kb = 180
    T = 20

    pred = torch.randn(B, Ka, T).float().to(device)
    targ = torch.randn(B, Ka, T).float().to(device)
    assign_L = torch.randn(VL, Kb).float().to(device)
    assign_R = torch.randn(VR, Kb).float().to(device)
    
    criterion = refineBLM_loss(0.1, 0.1).to(device)
    loss, l1, l2, l3 = criterion(pred, targ, assign_L, assign_R)
    print(loss.detach().cpu())
    print(l1, l2, l3)
