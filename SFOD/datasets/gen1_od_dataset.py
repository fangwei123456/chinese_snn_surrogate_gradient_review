import os
import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

from os import listdir
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from dataloader.prophesee import dat_events_tools
from dataloader.prophesee import npy_events_tools

# modified from https://github.com/loiccordone/object-detection-with-spiking-neural-networks/blob/main/datasets/gen1_od_dataset.py

class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # Time per T, y scaling, x scaling
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        save_file_name = f"SFOD_gen1_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        save_file = os.path.join(args.path, save_file_name)
        
        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}")

        # if mode != 'train':
        #     save_file_name = \
        #         f"gen1_{mode}_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file = os.path.join(args.path, save_file_name)

        #     if os.path.isfile(save_file):
        #         self.samples = torch.load(save_file)
        #         print("File loaded.")
        #     else:
        #         data_dir = os.path.join(args.path, mode)
        #         self.samples = self.build_dataset(data_dir, save_file)
        #         torch.save(self.samples, save_file)
        #         print(f"Done! File saved as {save_file}")
        # else:  # Since the train is divided into 3 parts, it is processed separately.
        #     save_file_name_a = \
        #         f"gen1_train_a_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_name_b = \
        #         f"gen1_train_b_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_name_c = \
        #         f"gen1_train_c_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_a = os.path.join(args.path, save_file_name_a)
        #     save_file_b = os.path.join(args.path, save_file_name_b)
        #     save_file_c = os.path.join(args.path, save_file_name_c)

        #     if os.path.isfile(save_file_a) and os.path.isfile(save_file_b) and os.path.isfile(save_file_c):
        #         self.samples = torch.load(save_file_a)
        #         self.samples.extend(torch.load(save_file_b))
        #         self.samples.extend(torch.load(save_file_c))
        #         print("File loaded.")
        #     else:
        #         print('Processing train_a ...')
        #         data_dir = os.path.join(args.path, 'train_a')
        #         self.samples = self.build_dataset(data_dir, save_file_a)
        #         torch.save(self.samples, save_file_a)
        #         print(f"Done! File saved as {save_file_a}")
        #         self.samples = []

        #         print('Processing train_b ...')
        #         data_dir = os.path.join(args.path, 'train_b')
        #         self.samples = self.build_dataset(data_dir, save_file_b)
        #         torch.save(self.samples, save_file_b)
        #         print(f"Done! File saved as {save_file_b}")
        #         self.samples = []

        #         print('Processing train_c ...')
        #         data_dir = os.path.join(args.path, 'train_c')
        #         self.samples = self.build_dataset(data_dir, save_file_c)
        #         torch.save(self.samples, save_file_c)
        #         print(f"Done! File saved as {save_file_c}")

        #         self.samples.extend(torch.load(save_file_a))
        #         print("File train_a loaded.")
        #         self.samples.extend(torch.load(save_file_b))
        #         print("File train_b loaded.")

    def __getitem__(self, index):
        (coords, feats), target = self.samples[index]

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                 if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])

            samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video, b)) is not None])
            pbar.update(1)

        pbar.close()
        return samples

    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), targets)

    def create_targets(self, boxes):
        boxes_unstructured=structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32)
        contiguous_boxes = np.ascontiguousarray(boxes_unstructured)
        copied_boxes = np.copy(contiguous_boxes)
        torch_boxes = torch.from_numpy(copied_boxes)

        # keep only last instance of every object per target
        _, unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True)  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (torch_boxes[:, 3] - torch_boxes[:, 1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}

    def create_data(self, events):
        events['t'] -= events['t'][0]
        #print('eventsp=',events['p'])
        feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

        coords = torch.from_numpy(
            structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        polarity = events['p'].copy().astype(np.int8)
        polarity[events['p'] == 0] = -1
        tbin_feats = (polarity * (tbin_coords + 1))
        tbin_feats[tbin_feats > 0] -= 1
        tbin_feats += (tbin_coords + 1).max()

        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin).to(bool)

        return coords.to(torch.int16), feats

class GEN1DetectionDataset_v(Dataset):
    def __init__(self, args, mode="train"):
        self.path='/datasets/MLG/mdy'
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # Time per T, y scaling, x scaling
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        # save_file_name = f"SFOD_gen1_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        # save_file = os.path.join(args.path, save_file_name)

        save_file_name = f"SFOD_T{self.T}_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"  
        save_file = os.path.join(self.path, save_file_name)
        # save_file_name1 = f"SFOD_lab_T{self.T}_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        # save_file1 = os.path.join(self.path, save_file_name1)

        self.save_npy=os.path.join(self.path,f"SFOD_T{self.T}_{self.sample_size//1000}",mode)
        print('self.save_npy=',self.save_npy)
        if not isdir( self.save_npy):
            os.makedirs( self.save_npy)
        data_dir = os.path.join(args.path, mode)
        if os.path.isfile(save_file):
            print("Start samples loaded.")
            self.samples = torch.load(save_file)
            print("End samples loaded.")
        else:
            self.samples = self.build_dataset(data_dir, save_file)
        # self.samples,self.labels=self.createAllBBoxDataset(save_file,save_file1)
        
        # if os.path.isfile(save_file):
        #     self.samples = torch.load(save_file)
        #     print("File loaded.")
        # else:
        #     data_dir = os.path.join(args.path, mode)
        #     self.samples = self.build_dataset(data_dir, save_file)
        #     torch.save(self.samples, save_file)
        #     print(f"Done! File saved as {save_file}")

        # if mode != 'train':
        #     save_file_name = \
        #         f"gen1_{mode}_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file = os.path.join(args.path, save_file_name)

        #     if os.path.isfile(save_file):
        #         self.samples = torch.load(save_file)
        #         print("File loaded.")
        #     else:
        #         data_dir = os.path.join(args.path, mode)
        #         self.samples = self.build_dataset(data_dir, save_file)
        #         torch.save(self.samples, save_file)
        #         print(f"Done! File saved as {save_file}")
        # else:  # Since the train is divided into 3 parts, it is processed separately.
        #     save_file_name_a = \
        #         f"gen1_train_a_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_name_b = \
        #         f"gen1_train_b_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_name_c = \
        #         f"gen1_train_c_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        #     save_file_a = os.path.join(args.path, save_file_name_a)
        #     save_file_b = os.path.join(args.path, save_file_name_b)
        #     save_file_c = os.path.join(args.path, save_file_name_c)

        #     if os.path.isfile(save_file_a) and os.path.isfile(save_file_b) and os.path.isfile(save_file_c):
        #         self.samples = torch.load(save_file_a)
        #         self.samples.extend(torch.load(save_file_b))
        #         self.samples.extend(torch.load(save_file_c))
        #         print("File loaded.")
        #     else:
        #         print('Processing train_a ...')
        #         data_dir = os.path.join(args.path, 'train_a')
        #         self.samples = self.build_dataset(data_dir, save_file_a)
        #         torch.save(self.samples, save_file_a)
        #         print(f"Done! File saved as {save_file_a}")
        #         self.samples = []

        #         print('Processing train_b ...')
        #         data_dir = os.path.join(args.path, 'train_b')
        #         self.samples = self.build_dataset(data_dir, save_file_b)
        #         torch.save(self.samples, save_file_b)
        #         print(f"Done! File saved as {save_file_b}")
        #         self.samples = []

        #         print('Processing train_c ...')
        #         data_dir = os.path.join(args.path, 'train_c')
        #         self.samples = self.build_dataset(data_dir, save_file_c)
        #         torch.save(self.samples, save_file_c)
        #         print(f"Done! File saved as {save_file_c}")

        #         self.samples.extend(torch.load(save_file_a))
        #         print("File train_a loaded.")
        #         self.samples.extend(torch.load(save_file_b))
        #         print("File train_b loaded.")

    def __getitem__(self, index):
        (coords, feats), target = torch.load(self.samples[index])

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                 if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        index=0
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])
            for b in boxes_per_ts:
                sample = self.create_sample(video, b)
                if sample is not None:
                    #samples.append(sample)
                    fn=  str(index)+'.npy'
                    npy_file=os.path.join(self.save_npy,fn)
                    torch.save(sample,npy_file)
                    samples.append(npy_file)
                    index+=1
            pbar.update(1)
        torch.save(samples, save_file)
        print(f"Done! File saved as {save_file}")

            #samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video, b)) is not None])
            #pbar.update(1)

        pbar.close()
        return samples

    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), targets)

    def create_targets(self, boxes):
        boxes_unstructured=structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32)
        contiguous_boxes = np.ascontiguousarray(boxes_unstructured)
        copied_boxes = np.copy(contiguous_boxes)
        torch_boxes = torch.from_numpy(copied_boxes)

        # keep only last instance of every object per target
        _, unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True)  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (torch_boxes[:, 3] - torch_boxes[:, 1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}

    def create_data(self, events):
        events['t'] -= events['t'][0]
        #print('eventsp=',events['p'])
        feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

        coords = torch.from_numpy(
            structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        polarity = events['p'].copy().astype(np.int8)
        polarity[events['p'] == 0] = -1
        tbin_feats = (polarity * (tbin_coords + 1))
        tbin_feats[tbin_feats > 0] -= 1
        tbin_feats += (tbin_coords + 1).max()

        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin).to(bool)

        return coords.to(torch.int16), feats



class GEN1DetectionDataset_(Dataset):
    def __init__(self, args, mode="train"):
        self.path='/datasets/MLG/mdy'
        self.root=args.path
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T

        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # Time per T, y scaling, x scaling
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        self.max_nr_bbox = 15
        self.nr_events_window = args.sample_size
      

        data_dir = os.path.join(self.root, mode)
        self.files = [os.path.join(data_dir, time_seq_name[:-9]) for time_seq_name in os.listdir(data_dir)
                        if time_seq_name[-3:] == 'npy']

        save_file_name = f"SFOD_T{self.T}_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"  
        save_file = os.path.join(self.path, save_file_name)
        save_file_name1 = f"SFOD_lab_T{self.T}_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        save_file1 = os.path.join(self.path, save_file_name1)

     
        self.save_npy=os.path.join(self.path,f"SFOD_T{self.T}_{self.sample_size//1000}",mode)
        print('self.save_npy=',self.save_npy)
        if not isdir( self.save_npy):
            os.makedirs( self.save_npy)
        self.samples,self.labels=self.createAllBBoxDataset(save_file,save_file1)

        self.nr_samples = len(self.files)
       

    def __getitem__(self, index):
        filename=self.samples[index]
        frames =torch.load(filename)
        
        label=self.labels[index]
       
        return frames, label
    
    def __len__(self):
        return len(self.samples)  

    def createAllBBoxDataset(self,save_file,save_file1):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
         unique indices file.
        """
        file_name_bbox_id = []
        sequence_starts=[]
        Frames=[]
        Lables=[]
        save_bbox_id_name = f"bbox_id_SFOD_T{self.T}_{self.sample_size//1000}_{self.mode}.pt"
        save_bbox_id = os.path.join(self.path, save_bbox_id_name)
        save_starts_name = f"starts_SFOD_T{self.T}_{self.sample_size//1000}_{self.mode}.pt"
        save_starts = os.path.join(self.path, save_starts_name)
        print('Building the Dataset index')
        if os.path.isfile(save_bbox_id):
            print("Start bbox_id loaded.")
            file_name_bbox_id = torch.load(save_bbox_id)
            sequence_starts=torch.load(save_starts)
            print("End bbox_id loaded.")
        else:
            pbar1 = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)
            for i_file, file_name in enumerate(self.files):
                bbox_file = os.path.join( file_name + '_bbox.npy')
                event_file = os.path.join(file_name + '_td.dat')
                f_bbox = open(bbox_file, "rb")
                start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
                dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
                f_bbox.close()
                unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)
                for unique_time in unique_ts:
                    sequence_start = self.searchEventSequence(event_file, unique_time, nr_window_events=self.nr_events_window)
                    sequence_starts.append(sequence_start)

                file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]
                pbar1.update(1)
            pbar1.close()
            torch.save(file_name_bbox_id, save_bbox_id )
            print(f"Done! File saved as {save_bbox_id }")
            torch.save(sequence_starts, save_starts )
            print(f"Done! File saved as {save_starts}")
        
        if os.path.isfile(save_file):
            print("Start events File loaded.")
            Frames = torch.load(save_file)
            print("End events File loaded.")
        else:
            print('Building the events Dataset rep')
            pbar= tqdm.tqdm(total=len(file_name_bbox_id), unit='File', unit_scale=True)
    
            for index in range(len(file_name_bbox_id)):
                event_file = os.path.join( file_name_bbox_id[index][0] + '_td.dat')
                events = self.readEventFile(event_file, sequence_starts[index],  nr_window_events=self.nr_events_window)
                frames=self.create_data(events)
                '''
                transform = Compose([ToFrame(sensor_size=[304,240,2],n_time_bins=self.T * self.tbin,),ToBinaRep(n_frames=self.T, n_bits=self.tbin),])
                if his:
                    frames_BR = transform(events)
                    histogram=self.generate_event_histogram(events, [240,304])
                    histogram_normalized=(histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram))
                    histogram_normalized=histogram_normalized.transpose(2, 0, 1)
                
                    histogram_normalized = np.expand_dims(histogram_normalized, axis=0)  # 在axis=0处插入新的维度
                    histogram_normalized = np.repeat(histogram_normalized, self.T, axis=0)  # 沿着axis=0进行复制
                    # print('frames.shape=',frames.shape)
                    # print('histogram_normalized.shape=',histogram_normalized.shape)

                    frames = np.concatenate((frames_BR, histogram_normalized), axis=1)
                    # print('fff.shape=',fff.shape)
                    # img_pos_normalized = fff[0,2,:,:] 
                    # img_pos_normalized *= 255
                    # img_pos_normalized = img_pos_normalized.astype(np.uint8)
                    # img_neg_normalized=fff[0,3,:,:] 
                    # img_neg_normalized *= 255
                    # img_neg_normalized = img_neg_normalized.astype(np.uint8)

                    # color_image_p = cv2.cvtColor( img_pos_normalized, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(f"/home/admin1/Ma/FSNNDet_830/results/{index}_histogram_0.jpeg", color_image_p)

                    # color_image_n = cv2.cvtColor( img_neg_normalized, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(f"/home/admin1/Ma/FSNNDet_830/results/{index}_histogram_1.jpeg", color_image_n)
                else:
                    frames = transform(events) 
                '''

                fn=  file_name_bbox_id[index][0].split('/')[-1]+'_'+str(file_name_bbox_id[index][1])+'.npy'
                npy_file=os.path.join(self.save_npy,fn)
                torch.save(frames,npy_file)
                Frames.append(npy_file)
                pbar.update(1)
            pbar.close()
            torch.save(Frames, save_file)
            print(f"Done! File saved as {save_file}")

        if os.path.isfile(save_file1):
            print("Start lables File loaded.")
            Lables = torch.load(save_file1)
            print("End lables File loaded.")
        else:
            print(f'Building the Dataset lab rep_mod_{self.mode}')
            pbar= tqdm.tqdm(total=len(file_name_bbox_id), unit='File', unit_scale=True)
            for index in range(len(file_name_bbox_id)):
                bbox_file = os.path.join( file_name_bbox_id[index][0] + '_bbox.npy')
                event_file = os.path.join(file_name_bbox_id[index][0] + '_td.dat')
                events = self.readEventFile(event_file, sequence_starts[index], nr_window_events=self.nr_events_window)
                f_bbox = open(bbox_file, "rb")
                start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
                dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
                f_bbox.close()
                unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)
                nr_unique_ts = unique_ts.shape[0]
                bbox_time_idx = file_name_bbox_id[index][1]
                if bbox_time_idx == (nr_unique_ts - 1):
                    end_idx = dat_bbox['ts'].shape[0]
                else:
                    end_idx = unique_indices[bbox_time_idx+1]
                bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]
                np_bbox = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5]]
                np_bbox = self.cropToFrame(np_bbox)
                torch_boxes=np_bbox[:,:4]
                torch_boxes[:, 2:] += torch_boxes[:, :2] # implicit conversion to xyxy
                torch_labels=np_bbox[:,4:]
                '''
                const_size_bbox = np.zeros([self.max_nr_bbox, 5])
                const_size_bbox[:np_bbox.shape[0], :] = np_bbox

                batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
                batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
                batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
                batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
                BBoxes = np.array(torch_boxes[:, :4], dtype=np.float32)
                torch_boxes[:, [0, 2]] = torch_boxes[:, [0, 2]]+self.ww
                torch_boxes[:, [1, 3]] = torch_boxes[:, [1, 3]]+self.hw
                if len(torch_boxes) != 0:
                    boxes = np.array(torch_boxes[:, :4],dtype=np.float32)
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.iw * self.output_shape[1], 0, self.output_shape[1] - 1)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.ih * self.output_shape[0], 0, self.output_shape[0] - 1)
                for i in range(len(boxes)):
                    bbox    = boxes[i]
                    Bbox = BBoxes[i]
                    cls_id  = int(torch_labels[i])
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h > 0 and w > 0:
                        radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        alpha=0.54
                        h_radius=int(h/2. *alpha)
                        w_radius=int(w/2. *alpha)

                        # ct_div = self.box_sum_events(events, Bbox[0], Bbox[2], Bbox[1], Bbox[3])
                        # ct_div = 0.5 * np.exp(-(ct_div * ct_div) / (2 * 10000 * 10000))
                        # print('1ct_div=', ct_div)
                   
                        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                        #batch_hm[:, :, cls_id] = self.draw_gaussian_T(batch_hm[:, :, cls_id], ct_int, h_radius,w_radius)
                        batch_hm[:, :, cls_id] = self.draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                        batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                        batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                        batch_reg_mask[ct_int[1], ct_int[0]] = 1
                '''
                Lables.append({'boxes': torch_boxes, 'labels': torch_labels})
                pbar.update(1)
            pbar.close()
       
            torch.save(Lables, save_file1)
            print(f"Done! File saved as {save_file1}")
   
        
        return Frames,Lables

    def create_data(self, events):
        #events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)
        events = events[:, [2, 1, 0, 3]]
        print('events.shape=',events.shape)
        events[:,0] -= events[:,0][0]
        #print('eventsp=',events[:,3])
        feats = torch.nn.functional.one_hot(torch.from_numpy(events[:,3]).to(torch.long), self.C)
        
        coords = torch.from_numpy(events[:, :3])
        print('coords.shape=',coords.shape)
        
        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        
        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events[:,0] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        polarity = events[:,3].copy().astype(np.int8)
        polarity[events[:,3] == 0] = -1
        tbin_feats = (polarity * (tbin_coords + 1))
        tbin_feats[tbin_feats > 0] -= 1
        tbin_feats += (tbin_coords + 1).max()

        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin).to(bool)
        
        coords=coords.to(torch.int16)
        print('feats.to(torch.float32).shape=',feats.to(torch.float32).shape)
        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        return sample

    def searchEventSequence(self, event_file, bbox_time, nr_window_events=100000):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        term_criterion = nr_window_events // 2
        nr_events = dat_events_tools.count_events(event_file)
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        low = 0
        high = nr_events

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            # self.seek_event(file_handle, middle)
            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=1)["ts"][0]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                file_handle.seek(ev_start + (middle - (term_criterion // 2) * ev_size))
                break

        file_handle.close()
        # we now know that it is between low and high
        return ev_start + low * ev_size 

    def readEventFile(self, event_file, file_position, nr_window_events=100000):
        file_handle = open(event_file, "rb")
        # file_position = ev_start + low * ev_size
        file_handle.seek(file_position)
        dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=nr_window_events)
        #print('dat_event=',dat_event)
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        # print('dat_event["_"]=',dat_event["_"])
        # print('x=',x)
        y = np.right_shift(
            np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        #p[p == 0] = -1
        events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)
        #print('events_np',events_np)

        return events_np     

    def generate_event_histogram(self, events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(int)
        y = y.astype(int)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

        return histogram 
    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        array_width = np.ones_like(np_bbox[:, 0]) * self.w - 1
        array_height = np.ones_like(np_bbox[:, 1]) * self.h - 1

        np_bbox[:, :2] = np.maximum(np_bbox[:, :2], np.zeros_like(np_bbox[:, :2]))
        np_bbox[:, 0] = np.minimum(np_bbox[:, 0], array_width)
        np_bbox[:, 1] = np.minimum(np_bbox[:, 1], array_height)

        np_bbox[:, 2] = np.minimum(np_bbox[:, 2], array_width - np_bbox[:, 0])
        np_bbox[:, 3] = np.minimum(np_bbox[:, 3], array_height - np_bbox[:, 1])

        return np_bbox
    
    def draw_gaussian(self,heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
           np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    
    def gaussian2D(self,shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
    
    def draw_gaussian_T(self,heatmap, center, h_radius, w_radius, k=1):
        h = 2 * h_radius + 1
        w =2 * w_radius+ 1
        sigma_x=w/6
        sigma_y=h/6

        gaussian = self.gaussian2D_T((h, w), sigma_x=sigma_x,sigma_y=sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
           np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D_T(self,shape, sigma_x=1,sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x)+y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self,det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def box_sum_events(self, events, x1, x2, y1, y2):
        # print('events=',events)
        # print('x1=',x1)
        # print('x2=', x2)
        # print('y1=', y1)
        # print('y2=', y2)
        # arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        # con=(arr[:,1]>=5)&(arr[:,1]<=10)&(arr[:,2]>=6)&(arr[:,2]<=15)
        # rrr=np.where(con)
        # print('rrr=',rrr)

        condition = (events[:, 0] >= x1) & (events[:, 0] <= x2) & (events[:, 1] >= y1) & (events[:, 1] <= y2)
        result = np.where(condition)
        # print('result=',result)
        sum_events = len(result[0])
        # print('sum_events=',sum_events)
        return sum_events
