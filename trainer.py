import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_loader import ObjectDataset
from torchvision_lib import utils, transforms as T
from torchvision_lib.engine import train_one_epoch, evaluate


class Trainer:
    def __init__(self, train_dir, test_dir, limit_to_cpu=False):
        self.device = self.choose_device(limit_to_cpu)
        self.model = self.get_model()
        self.model.to(self.device)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def choose_device(self, limit_to_cpu):
        if limit_to_cpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_model(self):
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2  # 6 class (buses) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def get_transform(self, is_train):
        transforms = []
        transforms.append(T.ToTensor())
        if is_train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def train(self, num_epochs, batch_size, add_text):
        # use our dataset and defined transformations
        dataset = ObjectDataset(self.train_dir, self.get_transform(is_train=True))
        dataset_test = ObjectDataset(self.test_dir, self.get_transform(is_train=False))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's is_train it for num_ephocs epochs
        for epoch in range(num_epochs):
            # is_train for one epoch, printing every num_epochs iterations
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_test, device=self.device)
            # if epoch % 10 == 0:
            # save each epoch
            self.save_model(epoch, batch_size, add_text)
        print("That's it trained!")

        return self.model, dataset_test

    def save_model(self, num_epochs, batch_size, add_text):
        import datetime
        from shutil import copyfile
        import os

        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        unique_filename = f"model__num_epochs-{num_epochs}__batch_size-{batch_size}__add_text={add_text}__{timestamp}.pt"
        unique_filepath = os.path.join(models_dir, unique_filename)
        standard_filepath = "model_state_dict.pt"

        torch.save(self.model.state_dict(), unique_filepath)
        copyfile(unique_filepath, standard_filepath)

        print("updated model_state_dict.pt")
        print(f" also saved to {unique_filepath}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs')
    parser.add_argument('--limit_to_cpu')
    parser.add_argument('--batch_size')
    parser.add_argument('--add_text')
    parser.add_argument('--train_dir')
    parser.add_argument('--test_dir')
    args = parser.parse_args()

    # parse cli arguments or take default values
    num_epochs = int(args.num_epochs)
    if num_epochs is None:
        print("USAGE python trainer.py --num_epochs <num>")
        exit(0)
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 2

    if args.add_text:
        add_text = args.add_text
    else:
        add_text = ""

    if args.limit_to_cpu is None:
        limit_to_cpu = False
    else:
        limit_to_cpu = True

    if args.train_dir is None:
        raise Exception("add --train_dir <train_dir_path>")
    if args.test_dir is None:
        raise Exception("add --test_dir <test_dir_path>")

    train_dir = args.train_dir #'data/doors/naor/train'
    test_dir = args.test_dir #'data/doors/naor/test'

    trainer = Trainer(train_dir, test_dir, limit_to_cpu=limit_to_cpu)
    model, dataset_test = trainer.train(num_epochs=num_epochs, batch_size=batch_size, add_text=add_text)
    print("DONE training!")
