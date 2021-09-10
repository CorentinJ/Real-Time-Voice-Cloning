import os 
import torch
import pickle  
import numpy as np

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager  

from torch.utils import data
from torch.utils.data.sampler import Sampler


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, feat_dir, mode):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.mode = mode
        self.step = 20
        self.split = 0
        
        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        manager = Manager()
        meta = manager.list(meta)
       
        dataset = manager.list(len(meta)*[None])  # <-- can be shared between processes.
        
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i,mode))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # self.load_data(meta, dataset, 0, mode)
            
        
        # very important to do dataset = list(dataset)            
        if mode == 'train':
            self.train_dataset = list(dataset)
            self.num_tokens = len(self.train_dataset)
        elif mode == 'test':
            self.test_dataset = list(dataset)
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError
        
        print('Finished loading {} dataset...'.format(mode))
        
        
        
    def load_data(self, submeta, dataset, idx_offset, mode):
        count = 0  
        for k, sbmt in enumerate(submeta):
            uttrs_list = len(sbmt[2]) * [None]
            uttrs = 3*[None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]
            

            # check that there are as many embeddings as speakers
            assert(len(sbmt[2]) == len(sbmt[1]))
            # fill in data
            for i, (embedding, speaker_save) in enumerate(zip(sbmt[1], sbmt[2])):
                sp_tmp = np.load(os.path.join(self.root_dir, speaker_save + ".npy"))
                f0_tmp = np.load(os.path.join(self.feat_dir, speaker_save + ".npy"))
                if self.mode == 'train':
                    sp_tmp = sp_tmp[self.split:, :]
                    f0_tmp = f0_tmp[self.split:]
                elif self.mode == 'test':
                    sp_tmp = sp_tmp[:self.split, :]
                    f0_tmp = f0_tmp[:self.split]
                else:
                    raise ValueError

                uttrs[1] = embedding
                uttrs[2] = ( sp_tmp, f0_tmp )
                uttrs_list[i] = uttrs

            dataset[idx_offset+k] = uttrs_list
            # assert(not isinstance(uttrs_list, None))
            # dataset[count] = uttrs
            # count += 1
            
                   
        
    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        list_uttrs = dataset[index]
        
        # pick random uttr:
        rand_uttr = np.random.randint(len(list_uttrs))
        list_uttrs = list_uttrs[rand_uttr]

        spk_id_org = list_uttrs[0]
        emb_org = list_uttrs[1]
        melsp, f0_org = list_uttrs[2]
        
        return spk_id_org, melsp, emb_org, f0_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    

class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        
    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        new_batch = []
        for token in batch:
            spk_id, aa, b, c = token
            len_crop = np.random.randint(min(self.min_len_seq, len(aa)-1), min(self.max_len_seq+1, len(aa)), size=2) # 1.5s ~ 3s
            
            left = np.random.randint(0, len(aa)-len_crop[0], size=2)
            
            a = aa[left[0]:left[0]+len_crop[0], :]
            c = c[left[0]:left[0]+len_crop[0]]
            
            a = np.clip(a, 0, 1)
            
            a_pad = np.pad(a, ((0,self.max_len_pad-a.shape[0]),(0,0)), 'constant')
            c_pad = np.pad(c[:,np.newaxis], ((0,self.max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (spk_id, a_pad, b, c_pad, len_crop[0]) ) 
            
        batch = new_batch  
        
        spkr, a, b, c, d = zip(*batch)
        melsp = torch.from_numpy(np.stack(a, axis=0))
        spk_emb = torch.from_numpy(np.stack(b, axis=0))
        pitch = torch.from_numpy(np.stack(c, axis=0))
        len_org = torch.from_numpy(np.stack(d, axis=0))
        
        return spkr, melsp, spk_emb, pitch, len_org
    


    
class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    """
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)        
    
    
    

def get_loader(hparams):
    """Build and return a data loader."""
    
    dataset = Utterances(hparams.root_dir, hparams.feat_dir, hparams.mode)
    
    my_collator = MyCollator(hparams)

    # Fix randomization random_split
    torch.manual_seed(0)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    sample_size = 4
    test_size -= 4
    sample_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [sample_size,  test_size])

    # Set randomization back
    torch.manual_seed(torch.initial_seed())
    
    sampler_v = MultiSampler(len(test_dataset), hparams.samplier, shuffle=hparams.shuffle)
    sampler_t = MultiSampler(len(train_dataset), hparams.samplier, shuffle=hparams.shuffle)
    sampler_s = MultiSampler(len(sample_dataset), 1, shuffle=hparams.shuffle)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    
    

    data_loader_val = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler_v,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)

    data_loader_train = data.DataLoader(dataset=train_dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler_t,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)
    
    data_loader_samp = data.DataLoader(dataset=sample_dataset,
                                  batch_size=1,
                                  sampler=sampler_s,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)

    return [data_loader_train, data_loader_val, data_loader_samp]