import torch
import numpy as np
import torch.nn as nn
import nibabel as nib


class refineBLM(nn.Module):

    def __init__(self, blm, device='cuda'):
        super(refineBLM, self).__init__()

        self.BLM = blm

        label_A424 = nib.load('datasets/A424.dlabel.nii')
        label_A424 = label_A424.get_fdata().astype(int).squeeze()
        print(label_A424.shape)

        assign_R, assign_L, assign_sub = [], [], []
        ind_R, ind_L = (label_A424 >= 1) & (label_A424 <= 180), (label_A424 >= 181) & (label_A424 <= 360)
        ind_sub = (label_A424 > 360)

        for ind in range(1, 181):
            assign_R.append(label_A424[ind_R] == ind)
        assign_R = np.stack(assign_R, axis=1)
        print(assign_R.shape, assign_R.sum(0))
        assign_R = 10.0 * assign_R - 10.0 * (~assign_R)

        for ind in range(181, 361):
            assign_L.append(label_A424[ind_L] == ind)
        assign_L = np.stack(assign_L, axis=1)
        print(assign_L.shape, assign_L.sum(0))
        assign_L = 10.0 * assign_L - 10.0 * (~assign_L)

        for ind in range(361, 425):
            assign_sub.append(label_A424[ind_sub] == ind)
        assign_sub = np.stack(assign_sub, axis=1)
        print(assign_sub.shape, assign_sub.sum(0))

        self.assign_L = nn.Parameter(torch.from_numpy(assign_L.astype(np.float32)))  # (V, K)
        self.assign_R = nn.Parameter(torch.from_numpy(assign_R.astype(np.float32)))

        self.assign_sub = torch.from_numpy(assign_sub.astype(np.float32)).to(device)
        self.assign_sub_sum = self.assign_sub.sum(dim=0, keepdims=True)[None, :, :]

        self.ind_L = torch.from_numpy(ind_L).to(device)
        self.ind_R = torch.from_numpy(ind_R).to(device)
        self.ind_sub = torch.from_numpy(ind_sub).to(device)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, timeseries):
        """
        :param timeseries:  (B, T, 91k)
        :return: parceled: (B,T,424), predict: (B, T2, 424)
        """
        soft_R = self.softmax(self.assign_R)
        soft_L = self.softmax(self.assign_L)
        # print(soft_L)

        parcel_ts_R = torch.einsum('b t v, v k -> b t k', timeseries[:, :, self.ind_R], soft_R) / soft_R.sum(dim=0, keepdims=True)[None, :, :]
        print(parcel_ts_R.shape)  # (B, T, 180)

        parcel_ts_L = torch.einsum('b t v, v k -> b t k', timeseries[:, :, self.ind_L], soft_L) / soft_L.sum(dim=0, keepdims=True)[None, :, :]
        print(parcel_ts_L.shape)  # (B, T, 180)

        parcel_ts_sub = torch.einsum('b t v, v k -> b t k', timeseries[:, :, self.ind_sub], self.assign_sub) / self.assign_sub_sum
        print(parcel_ts_sub.shape)  # (B, T, 180)

        parcel_ts = torch.concat([parcel_ts_R, parcel_ts_L, parcel_ts_sub], dim=2)
        print(parcel_ts.shape)  # (B, T, 424)

        parcel_ts = (parcel_ts - parcel_ts.mean(axis=1, keepdims=True)) / parcel_ts.std(axis=1, keepdims=True)

        return self.BLM(parcel_ts), soft_L, soft_R


if __name__ == '__main__':
    device = 'cuda'
    B, tsd = 8, 91282
    T = 200

    model = refineBLM(None).to(device)

    ts = torch.randn(B, T, tsd).float().to(device)
    model(ts)
