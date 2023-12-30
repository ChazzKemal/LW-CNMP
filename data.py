import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
PI = 3.141592653589793


class CNPDemonstrationDataset:
    def __init__(self) -> None:
        self.N = 0
        self.data = []

    def get_sample(self, batch_size=1, max_context=10, max_target=10):
        context_all, target_all, context_mask, target_mask = [], [], [], []
        for _ in range(batch_size):
            n_context = torch.randint(1, max_context, ())
            n_target = torch.randint(1, max_target, ())
            idx = torch.randint(0, self.N, ())
            traj = self.data[idx]
            R = torch.randperm(traj.shape[0])
            context = traj[R[:n_context]]
            target = traj[R[:(n_context+n_target)]]
            context_all.append(context)
            target_all.append(target)
            context_mask.append(torch.ones(context.shape[0]))
            target_mask.append(torch.ones(target.shape[0]))
        context_all = pad_sequence(context_all, batch_first=True)
        target_all = pad_sequence(target_all, batch_first=True)
        context_mask = pad_sequence(context_mask, batch_first=True)
        target_mask = pad_sequence(target_mask, batch_first=True)
        return context_all, target_all, context_mask, target_mask
   
    def get_sample_VQ(self, batch_size=2, max_context=10, max_target=10):
        context_all, target_all, context_mask, target_mask, source_labels = [], [], [], [], []
       
        half_batch_size = batch_size // 2
       
        # Helper function to sample trajectories
        def sample_trajectories(data_list, n_samples, source_label):
            for _ in range(n_samples):
                n_context = torch.randint(1, max_context, ())
                n_target = torch.randint(1, max_target, ())
                idx = torch.randint(0, len(data_list), ())
                traj = data_list[idx]
                R = torch.randperm(traj.shape[0])
                context = traj[R[:n_context]]
                target = traj[R[:(n_context+n_target)]]
                context_all.append(context)
                target_all.append(target)
                context_mask.append(torch.ones(context.shape[0]))
                target_mask.append(torch.ones(target.shape[0]))
                source_labels.append(source_label)  # Append the source label
        
        # Sample from data1
        sample_trajectories(self.data[0], half_batch_size, 0)
        
        # Sample from data2
        sample_trajectories(self.data[1], half_batch_size, 1)
        
        context_all = pad_sequence(context_all, batch_first=True)
        target_all = pad_sequence(target_all, batch_first=True)
        context_mask = pad_sequence(context_mask, batch_first=True)
        target_mask = pad_sequence(target_mask, batch_first=True)
        source_labels = torch.tensor(source_labels)
        
        return context_all, target_all, context_mask, target_mask, source_labels



#less demos, with primitive information inside the data i have changed the task parameter with -1
class GridDataset_tp(CNPDemonstrationDataset):
    def __init__(self):
        self.data = []
        for i in range(5):
            for _ in range(100):
                y = -2 + i
                p1 = torch.rand(2) * 0.3 - torch.tensor([2, y])
                p2 = torch.rand(2) * 0.3 - torch.tensor([0.75, y])
                p3 = torch.rand(2) * 0.3 + torch.tensor([0.75, -y])
                p4 = torch.rand(2) * 0.3 + torch.tensor([2, -y])
                curve = bezier([p1, p2, p3, p4], 200)
                curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), torch.ones(curve.shape[0], 1)*(-1) , curve], dim=-1)
                self.data.append(curve)

        for i in range(5):
            for _ in range(100):
                x = -2 + i
                p1 = torch.rand(2) * 0.3 - torch.tensor([x, 2])
                p2 = torch.rand(2) * 0.3 - torch.tensor([x, 0.75])
                p3 = torch.rand(2) * 0.3 + torch.tensor([-x, 0.75])
                p4 = torch.rand(2) * 0.3 + torch.tensor([-x, 2])
                curve = bezier([p1, p2, p3, p4], 200)
                curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), torch.ones(curve.shape[0], 1) , curve], dim=-1)
                self.data.append(curve)

        self.N = len(self.data)


class TwoSkillsDataset(CNPDemonstrationDataset):
    def __init__(self, N, freq_shift_noise=0.2, amp_shift_noise=0.1, val_split=0.2):
        self.N = int(N*(1-val_split))
        b = 1 - freq_shift_noise/2

        x = torch.linspace(0, PI*2, 200).repeat(N//2, 1) * (torch.rand(N//2, 1) * freq_shift_noise + b) + \
            torch.randn(N//2, 1)*0.03
        y = torch.sin(x)*0.2 + torch.randn(N//2, 1) * amp_shift_noise
        data = torch.stack([x, y], dim=-1)
        x = torch.linspace(0, PI*2, 200).repeat(N//2, 1) * (torch.rand(N//2, 1) * freq_shift_noise + b) + \
            torch.randn(N//2, 1)*0.03
        y = torch.sin(x)*0.2 + torch.randn(N//2, 1) * amp_shift_noise
        data = torch.cat([data, torch.stack([y.flip(1), x.flip(1)], dim=-1)], dim=0)
        data = torch.cat([torch.linspace(0, 1, 200).repeat(data.shape[0], 1).unsqueeze(2), data], dim=-1)
        self.data = data

# completely linear demonstrations
class GridDataset_linear(CNPDemonstrationDataset):
    def __init__(self, divider=1):
        self.data = []
        
        # Function to create a linear curve between two points
        def linear_curve(p1, p2, num_points=200):
            t = torch.linspace(0, 1, num_points).unsqueeze(1)
            curve = (1 - t) * p1 + t * p2
            return curve
        
        for i in range(5):
            for _ in range(2):
                for j in range(10):
                    for k in range(1, (11-j)):
                        y = -2 + i
                        
                        p1 = -torch.tensor([2-j*0.4, y])
                        p2 = -torch.tensor([2-(j+k)*0.4, y])
                        
                        curve = linear_curve(p1, p2)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        self.data.append(curve)

        for i in range(5):
            for _ in range(2):
                for j in range(10):
                    for k in range(1, (11-j)):
                        x = -2 + i
                        
                        p1 = -torch.tensor([x, 2-j*0.4])
                        p2 = -torch.tensor([x, 2-(j+k)*0.4])
                        
                        curve = linear_curve(p1, p2)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        self.data.append(curve)

        self.N = len(self.data)




# the bezier demonstrations, with varying length and curvature
class GridDataset(CNPDemonstrationDataset):
    def __init__(self,divider=1):
        self.data = []
        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):
                        y = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([2-j*0.4, y])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([0.75, y])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([0.75, -y])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([2-(j+k)*0.4, y])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        self.data.append(curve)

        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):            
                        x = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([x, 2-j*0.4])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([x, 0.375])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([-x, 0.75])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([x, 2-(j+k)*0.4])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        self.data.append(curve)

        self.N = len(self.data)

#grid dataset divided to two primitives
class GridDatasetV2(CNPDemonstrationDataset):
    def __init__(self,divider=1):
        self.data = []
        data1=[]
        data2=[]
        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):
                        y = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([2-j*0.4, y])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([0.75, y])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([0.75, -y])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([2-(j+k)*0.4, y])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1),curve], dim=-1)
                        data1.append(curve)

        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):            
                        x = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([x, 2-j*0.4])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([x, 0.375])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([-x, 0.75])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([x, 2-(j+k)*0.4])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1),torch.zeros(curve.shape[0], 1) ,curve], dim=-1)
                        data2.append(curve)

        self.N = len(data1)+ len(data2)
        self.data.append(data1)
        self.data.append(data2)


class GridDatasetV2_tp(CNPDemonstrationDataset):
    def __init__(self,divider=1):
        self.data = []
        data1=[]
        data2=[]
        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):
                        y = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([2-j*0.4, y])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([0.75, y])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([0.75, -y])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([2-(j+k)*0.4, y])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), torch.zeros(curve.shape[0], 1) , curve], dim=-1)
                        data1.append(curve)

        for i in range(5):
            for _ in range(2):
                for j in range (10):
                    for k in range(1,(11-j)):            
                        x = -2 + i
                        p1 = torch.rand(2) * 0.3 - torch.tensor([x, 2-j*0.4])
                        p2 = torch.rand(2) * 0.3 - torch.tensor([x, 0.375])
                        p3 = torch.rand(2) * 0.3 + torch.tensor([-x, 0.75])
                        p4 = torch.rand(2) * 0.3 - torch.tensor([x, 2-(j+k)*0.4])
                        curve = bezier([p1, p2, p3, p4], 200)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), torch.ones(curve.shape[0], 1) , curve], dim=-1)
                        data2.append(curve)

        self.N = len(data1)+ len(data2)
        self.data.append(data1)
        self.data.append(data2)



class GridDatasetV2_linear(CNPDemonstrationDataset):
    def __init__(self, divider=1):
        self.data = []
        data1 = []
        data2 = []
        
        # Function to create a linear curve between two points
        def linear_curve(p1, p2, num_points=200):
            t = torch.linspace(0, 1, num_points).unsqueeze(1)
            curve = (1 - t) * p1 + t * p2
            return curve

        for i in range(5):
            for _ in range(2):
                for j in range(10):
                    for k in range(1, (11-j)):
                        y = -2 + i
                        
                        p1 = -torch.tensor([2-j*0.4, y])
                        p2 = -torch.tensor([2-(j+k)*0.4, y])
                        
                        curve = linear_curve(p1, p2)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        data1.append(curve)

        for i in range(5):
            for _ in range(2):
                for j in range(10):
                    for k in range(1, (11-j)):
                        x = -2 + i
                        
                        p1 = -torch.tensor([x, 2-j*0.4])
                        p2 = -torch.tensor([x, 2-(j+k)*0.4])
                        
                        curve = linear_curve(p1, p2)
                        curve = torch.cat([torch.linspace(0, 1, curve.shape[0]).reshape(-1, 1), curve], dim=-1)
                        data2.append(curve)

        self.N = len(data1) + len(data2)
        self.data.append(data1)
        self.data.append(data2)




class BaxterDemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_context=10, max_target=10):
        data_dict = torch.load(path)
        self.data = []
        self.names = []
        for key, value in data_dict.items():
            self.data.append(value)
            self.names.append(key)
        self.max_context = max_context
        self.max_target = max_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        n_context = torch.randint(3, self.max_context, ())
        n_target = torch.randint(3, self.max_target, ())
        R = torch.randperm(traj.shape[0])
        context = traj[R[:n_context]]
        target = traj[R[:(n_context+n_target)]]
        return context, target


def unequal_collate(batch):
    context_all, target_all, context_mask, target_mask = [], [], [], []
    for context, target in batch:
        context_all.append(context)
        target_all.append(target)
        context_mask.append(torch.ones(context.shape[0]))
        target_mask.append(torch.ones(target.shape[0]))
    context_all = pad_sequence(context_all, batch_first=True)
    target_all = pad_sequence(target_all, batch_first=True)
    context_mask = pad_sequence(context_mask, batch_first=True)
    target_mask = pad_sequence(target_mask, batch_first=True)
    return context_all, target_all, context_mask, target_mask


def bezier(p, steps=100):
    t = torch.linspace(0, 1, steps).reshape(-1, 1)
    curve = torch.pow(1-t, 3)*p[0] + 3*torch.pow(1-t, 2)*t*p[1] + 3*(1-t)*torch.pow(t, 2)*p[2] + torch.pow(t, 3)*p[3]
    return curve
