import os, time, copy
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
from AlexNet_me import NewLeNet, autoencoder
from WGAN import *

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)

class incClassifier():
    def __init__(self, useCuda=True):
        super(incClassifier).__init__()
        self.useCuda = useCuda
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mainNet = NewLeNet()
        if useCuda: self.mainNet = self.mainNet.cuda()
        self.weightshape = len(self.mainNet.currentWeights())
        self.SoftMax = nn.Softmax(dim=0)
        self.mataData = {
            'gate': [],
            'gateKey': [],
            'generator': [],
            'classifier': [],
            'classes': []
        }

    def _train_lenet_(self, model, criterion, dataloaders, optimizer, scheduler, resume, num_epochs=25):
        since = time.time()
        model_weights = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            start_epoch = start_epoch - 2

            model.load_state_dict(checkpoint['state_dict_model'])

            print('load')
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            torch.cuda.empty_cache()
            model_weights = checkpoint['model_weights']

            print(resume, checkpoint['arch'])
            del checkpoint
        else:
            start_epoch = 0
            print("=> no checkpoint found at '{}'".format(resume))

        for epoch in range(start_epoch, num_epochs):
            epoch_weights = []
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs
                    labels = labels
                    if self.useCuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = model(inputs)
                        outputs = model(inputs, phase == 'train')
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                        c_w = copy.deepcopy(model.currentWeights()).to('cpu')
                        epoch_weights.append(c_w)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_weights = [x for x in epoch_weights]

                epoch_file_name = resume
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'newLeNet',
                    'model': model,
                    'state_dict_model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_weights' : model_weights,
                }, epoch_file_name)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, model_weights

    def _train_encoder_(self, model, criterion, dataloaders, optimizer, num_epochs=25):

        print('training Autoencoder')

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    img, _ = data
                    img = img.view(img.size(0), -1)
                    img = Variable(img)
                    if self.useCuda: img = img.cuda()
                    # ===================forward=====================
                    output = model(img)
                    loss = criterion(output, img)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data))
        return model

    def _train_GAN_(self, generator, discriminator, dataloader, optimizer_G, optimizer_D, resume, num_epochs=100,n_critic=5, clip_value=0.01):
        since = time.time()

        latent_dim = generator.model[0].in_features

        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            start_epoch = start_epoch-2

            generator = checkpoint['generator']
            discriminator = checkpoint['discriminator']

            generator.load_state_dict(checkpoint['state_dict_generator'])
            discriminator.load_state_dict(checkpoint['state_dict_discriminator'])

            print('load')
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            torch.cuda.empty_cache()
            del checkpoint
        else:
            start_epoch = 0
            print("=> no checkpoint found at '{}'".format(resume))

        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            running_disc_loss = 0.0
            running_gen_loss = 0.0

            # Iterate over data.
            acc_l = []
            for i, (inputs) in enumerate(dataloader):
                # Configure input
                real_inputs = Variable(inputs)
                if self.useCuda: real_inputs = real_inputs.cuda()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.Tensor(np.random.normal(0, 1, (inputs.shape[0], latent_dim)))
                z = Variable(z)
                if self.useCuda: z = z.cuda()

                # Generate a batch of images
                fake_inputs = generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(real_inputs)) + torch.mean(discriminator(fake_inputs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # Train the generator every n_critic iterations
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_inputs = generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(discriminator(gen_inputs))

                    loss_G.backward()
                    optimizer_G.step()

                # -----------------------------------------------------------

                # statistics
                running_disc_loss += loss_D.data
                running_gen_loss += loss_G.data

            epoch_disc_loss = running_disc_loss / len(dataloader)
            epoch_gen_loss = running_gen_loss / len(dataloader)
            # epoch_acc_mainNet = sum(acc_l)/max(len(acc_l),1)

            print('Disc Loss : {}'.format(epoch_disc_loss))
            print('Gen Loss : {}'.format(epoch_gen_loss))
            # print('avgAcc on AlexNets: {}'.format(epoch_acc_mainNet))
            epoch_file_name = resume
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'WGAN',
                'generator': generator,
                'discriminator': discriminator,
                'state_dict_generator': generator.state_dict(),
                'state_dict_discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, epoch_file_name)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return generator, discriminator

    def __getavgLoss__(self, model, criterion, dataloader, size):
        Loss = 0
        for i, (images, _) in enumerate(dataloader):
            # flatten the image
            images = images.view(images.size(0), -1)
            if self.useCuda: images = images.cuda()
            out = model(images)
            Loss += criterion(out, images)

        Loss = Loss / size
        return Loss

    def _trainClassifier(self, dataloaders, num_classes, num_epochs=500, lr=0.001, resume='checkpoint/LeNet_epoch.pth'):

        num_ftrs = self.mainNet.classifier[4].in_features
        self.mainNet.classifier[4] = nn.Linear(num_ftrs, len(num_classes))

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = torch.optim.Adam(self.mainNet.parameters(), lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # resume = os.path.join('model', 'LeNet_epoch.pth.tar')
        if self.useCuda:
            self.mainNet.cuda()
        self.mainNet, model_weights = self._train_lenet_(self.mainNet, criterion, dataloaders, optimizer_ft,
                                                       exp_lr_scheduler, resume, num_epochs)
        return model_weights

    def _trainGateEncoder(self, dataloaders, num_epochs=20, lr=0.001, dim=100):
        encoder = autoencoder(32 * 32 * 3, dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        # iter_per_epoch = len(dataloaders['train'])+len(dataloaders['val'])
        # resume = os.path.join(check_dir, 'LeNetWGAN100_epoch.pth.tar')
        if self.useCuda:
            encoder.cuda()
        encoder = self._train_encoder_(encoder, criterion, dataloaders, optimizer, num_epochs=num_epochs)
        avgLoss = self.__getavgLoss__(encoder, criterion, dataloaders['train'], len(dataloaders['train']))
        return encoder, avgLoss

    def _traingenerator(self, task_weights, check_dir):
        lr = 1e-3
        n_critic = 5
        clip_value = 0.01
        num_epoch = 700
        insize = 100
        in_ft = len(task_weights[0])

        # check_dir = 'checkpoint'
        resume = os.path.join(check_dir, 'LeNetWGAN100_epoch.pth.tar')

        generator = Generator(insize, in_ft)
        discriminator = Discriminator(in_ft)
        if self.useCuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        generator, discriminator = self._train_GAN_(generator, discriminator, task_weights, optimizer_G, optimizer_D, resume, num_epoch)
        return generator

    def testing(self, dataloader):
        testLoss = 0
        test_corrects = 0
        criterion = nn.CrossEntropyLoss()
        for i, (images, labels) in enumerate(dataloader):
            if self.useCuda:
                images = images.cuda()
                labels = labels.cuda()

            with torch.set_grad_enabled(False):
                outputs = self.mainNet(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            testLoss += loss.item() * images.size(0)
            test_corrects += torch.sum(preds == labels.data)

        fin_loss = testLoss / len(dataloader.dataset)
        fin_acc = test_corrects.double() / len(dataloader.dataset)
        print('Loss: {:.4f} Acc: {:.4f}'.format(fin_loss, fin_acc))

        return fin_loss, fin_acc

    def __Relatedness__(self, Era, Erk):
        rel = 1-abs((Era-Erk)/Erk)
        return rel

    def _taskRelatedness(self, Erk):
        index = -1
        relations = []
        for Era in self.mataData['gateKey']:
            relations.append(self.__Relatedness__(Era, Erk))
        try :
            index = relations.index(min(relations))
            print('Most relevant Task at :', index)
        except Exception as er:
            print('No previous task')
        return index

    def _getBestModel(self, Erk):
        index = self._taskRelatedness(Erk)
        if index == -1: return self.mainNet
        generator = self.mataData['generator'][index]
        z = torch.Tensor(np.random.normal(0, 1, (self.weightshape, generator.model[0].in_features)))
        z = Variable(z)
        if self.useCuda: z = z.cuda()
        fake_inputs = generator(z).detach()
        self.mainNet.setmyweights(fake_inputs[-1])
        return self.mainNet

    def __relavant_task__(self, x):
        errors = []
        x = x.view(x.size(0), -1)
        x = Variable(x)
        criterion = nn.MSELoss()
        if self.useCuda: x = x.cuda()
        for gate in self.mataData['gate']:
            output = gate(x)
            loss = criterion(output, x)
            errors.append(loss)
        _, ind = self.SoftMax(torch.tensor(errors))
        return ind

    def __set_model__(self, index):
        generator = self.mataData['generator'][index]
        z = torch.Tensor(np.random.normal(0, 1, (self.weightshape, generator.model[0].in_features)))
        z = Variable(z)
        if self.useCuda: z = z.cuda()
        fake_inputs = generator(z).detach()
        self.mainNet.setmyweights(fake_inputs[-1])
        self.mainNet.classifier[4] = self.mataData['classifier'][index]
        pass

    def __revelentError__(self, ers, t):
        temp = torch.exp(-ers/t)
        return temp/torch.sum(ers)

    def __predict__(self, x):
        criterion = nn.MSELoss()
        er = []
        for gate in self.mataData['gate']:
            gate.eval()
            with torch.set_grad_enabled(False):
                _x = x.view(1, -1)
                er.append(criterion(gate(_x), _x).data)
        # ind = torch.max(self.SoftMax(torch.Tensor(er)), 0)[1]
        ind = torch.max(self.__revelentError__(torch.Tensor(er), 0.5), 0)[1]
        self.__set_model__(int(ind))
        self.mainNet.eval()
        with torch.set_grad_enabled(False):
            x = x.unsqueeze(0)
            output = self.mainNet(x)
        _, pred = torch.max(output, 1)
        return pred, ind, self.mataData['classes'][ind][int(pred)]

    def predict(self, images):
        pred_labs = []
        pred_tsks = []
        pred_clss = []
        if self.useCuda:
            images = images.cuda()
        for i, image in enumerate(images):
            lab, tsk, cls = self.__predict__(image)
            pred_labs.append(lab)
            pred_tsks.append(tsk)
            pred_clss.append(cls)
        pred_labs, pred_tsks = torch.Tensor(pred_labs), torch.Tensor(pred_tsks)
        if self.useCuda:
            pred_labs, pred_tsks = pred_labs.cuda(), pred_tsks.cuda()
        return pred_labs, pred_tsks, pred_clss

    def trainNewTask(self, datasets, dataloaders, labels, check_dir):
        dssize = len(datasets['train'])+len(datasets['val'])
        encoder, avgLoss = self._trainGateEncoder(dataloaders, num_epochs=100, lr=0.001, dim=100)
        self.mainNet = self._getBestModel(avgLoss)
        model_weights = self._trainClassifier(dataloaders, labels, num_epochs=250, lr=0.001, resume=check_dir+'/LeNet_epoch.pth')
        generator = self._traingenerator(model_weights, check_dir)

        self.mataData['gate'].append(encoder)
        self.mataData['gateKey'].append(avgLoss)
        self.mataData['classifier'].append(copy.deepcopy(self.mainNet.classifier[4]))
        self.mataData['classes'].append(labels)
        self.mataData['generator'].append(copy.deepcopy(generator))
        print('Done')
