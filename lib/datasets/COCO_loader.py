import torch.utils.data as data

import json

class CocoLoader(data.Dataset):
    def __init__(self):
        super(CocoLoader, self).__init__()
        data_json_dir = "/home/tusharkumar91/WS/MAttNet/cache/prepro/refcocog_umd/"
        data_json_file = "data.json"
        self.data_json_path = data_json_dir + data_json_file
        with open(self.data_json_path) as f:
            data_json = json.load(f)
        self.imgs = data_json["images"]

    def __getitem__(self, index):
        item = self.imgs[index]
        return {"item" : item["file_name"], "id" : item["image_id"]}

    def __len__(self):
        return len(self.imgs)

    def get_data_json_path(self):
        return self.data_json_path
