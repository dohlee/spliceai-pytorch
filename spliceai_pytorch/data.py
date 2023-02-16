from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    import torch
    import h5py
    
    h5f = h5py.File('../spliceai_train_code/Canonical/dataset_train_all.h5')
    idx = 1

    X, Y = h5f[f'X{idx}'][:], h5f[f'Y{idx}'][0, ...]
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8)

    for batch in loader:
        print(batch[0].shape, batch[1].shape)
        break