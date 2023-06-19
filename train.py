# -*- coding: utf-8 -*-
import time
import json
from predictnet import PResnet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
import matplotlib.pyplot as plt
classes = ('10000 ton ocean going ship', 'aerospace measurement and control ship', 'aircraft carrier', 'amphibious assault ship','air cushion landing craft',
                     'a sloop with a top sail and a bow sail', 'bakantin type', 'bamboo chops', 'barker type schooner', 'battle ship',
                     'brig', 'brigantin type', 'bulk carrier', 'canoe','cat-rigged boat','cruiser','destroyer','dragon boat','escort','fishing boats',
                      'full sail type', 'gaffsail schooner','high speed boat','icebreaker','industrial operation ship','landing ship','large passenger ship',
                     'leather valve ship', 'liquefied gas tanker', 'marine fire boat', 'military ship patrol boat','mine ship','minesweeper','missile ship','multihull ship',
                     'ocean going comprehensive survey ship','other ships','pilot ship','police ship','power boat','sailboard','sand carrier','search and rescue ship',
                     'sightseeing ship','single mast square sail','skona type','sloop schooner','small luxury yacht','submarine','submarine hunter','super large oil tanker','supply ship',
                     'the luxury cruise ship','tropedo boat','transport vessel','wave piercing catamaran','wooden pleasure boat','yawl','fpso')
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([
                                    transforms.Resize(510),
                                     transforms.RandomResizedCrop(448),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(510),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = datasets.ImageFolder(root="./train1", transform=data_transform["train"])
    train_num = len(train_dataset)
    bird_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in bird_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 24
    epochs = 80
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=8)
    validate_dataset = datasets.ImageFolder(root="./test1",transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=8)

    print("using {} images for training, {} images fot validation.".format(train_num, val_num))
    net = PResnet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.002,weight_decay=0.0005,momentum=0.9)

    best_acc = 0.0
    save_path = './model.pth'
    val_accuracy_list = []
    train_accuracy_list=[]
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        # train
        net.train()
        epoch_begin = time.time()
        if(epoch==20):
            optimizer = optim.SGD(net.parameters(), lr=0.001,weight_decay=0.0005,momentum=0.9)
        elif(epoch==30):
            optimizer = optim.SGD(net.parameters(), lr=0.0005,weight_decay=0.0005,momentum=0.9)
        elif(epoch==50):
            optimizer = optim.SGD(net.parameters(), lr=0.0001,weight_decay=0.0005,momentum=0.9)
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            batch_begin = time.time()
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            t = time.time() - batch_begin
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f},time:{:.4f}".format(epoch + 1,epochs,loss,float(t))
        t = time.time() - epoch_begin
        print("Epoch {} training ends, total {:.2f}s".format(epoch, t))
        # validate
        net.eval()
        val_acc = 0.0
        train_acc=0.0
        val_loss = 0.0
        train_loss=0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for train_data in train_bar:
                train_images,train_labels=train_data
                train_outputs=net(train_images.to(device))
                tmp_train_loss=loss_function(train_outputs,train_labels.to(device))
                train_predict=torch.max(train_outputs,dim=1)[1]
                train_acc+=torch.eq(train_predict, train_labels.to(device)).sum().item()
                train_loss+=tmp_train_loss.item()
                train_bar.desc = "valid in train_dataset epoch[{}/{}]".format(epoch + 1,epochs)
            
            for val_data in val_bar:
                val_images, val_labels = val_data
                print(val_labels)
                val_outputs = net(val_images.to(device))
                tmp_val_loss = loss_function(val_outputs, val_labels.to(device))
                val_predict = torch.max(val_outputs, dim=1)[1]
                print(val_predict)
                val_acc += torch.eq(val_predict, val_labels.to(device)).sum().item()
                val_loss+=tmp_val_loss.item()
                val_bar.desc = "valid in val_dataset epoch[{}/{}]".format(epoch + 1,epochs)

        train_accurate = train_acc / train_num
        val_accurate = val_acc / val_num

        if (val_accurate > best_acc):
            best_acc = val_accurate
        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f'
              % (epoch + 1, train_loss / train_num, train_accurate, val_loss / val_num, val_accurate))

        val_accuracy_list.append(val_accurate)
        train_accuracy_list.append(train_accurate)
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss / val_num)
        epochs_list.append(epoch + 1)
        torch.save(net.state_dict(), save_path)

        # train_acc && val_loss
        plt.figure()
        plt.plot(epochs_list, val_accuracy_list, color="red", label="val_acc")
        plt.plot(epochs_list, train_accuracy_list, color="green", label="train_acc")
        plt.xlabel("epochs")
        plt.ylabel("Acc")
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0, len(epochs_list), 20)])
        acc_gap = [i * 0.2 for i in range(0, min(int(len(epochs_list) / 2 + 1), 6))]
        acc_gap.append(max(val_accuracy_list))
        acc_gap.append(max(train_accuracy_list))
        plt.yticks(acc_gap)
        plt.grid()
        plt.legend()
        plt.savefig("Acc2.jpg")

        # train_loss && val_loss
        plt.figure()
        plt.plot(epochs_list, train_loss_list, color="red", label="train_loss")
        plt.plot(epochs_list, val_loss_list, color="green", label="val_loss")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0, len(epochs_list), 20)])
        plt.grid()
        plt.legend()
        plt.savefig("Loss.jpg")

    print('Finished Training')
    print("the best val_accuracy is : {}".format(best_acc))

if __name__ == '__main__':
    main()
