import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ScatteringDataset(Dataset):
    def __init__(self, root, nn_sp, nn_label, sp_range=(3, -4), lb_range=(3, -4)):
        self.root = root
        self.sp_path = os.path.join(self.root, nn_sp)
        self.label_path = os.path.join(self.root, nn_label)

        self.img_sp_path = sorted(os.listdir(self.sp_path),
                                  key=lambda x: int(x[sp_range[0]:sp_range[1]]))
        self.img_label_path = sorted(os.listdir(self.label_path),
                                     key=lambda x: int(x[lb_range[0]:lb_range[1]]))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            # torchvision.transforms.CenterCrop((256,256)),
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        sp_item_path = os.path.join(self.sp_path, self.img_sp_path[idx])
        lb_item_path = os.path.join(self.label_path, self.img_label_path[idx])

        sp_get = self.transform(Image.open(sp_item_path))
        lb_get = self.transform(Image.open(lb_item_path))
        return sp_get, lb_get

    def __len__(self):
        return len(self.img_sp_path)
