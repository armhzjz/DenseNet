import time
import numpy as np
import torch
import torch.nn as nn



def TrainModel(model, criterion, optimizer, nEpochs, train_stats=None, bestModelName=None, lr_update_at_Epoch_perc=0.20, minLr_val_at_Epoch_perc=0.9,
               train_loader=None, valid_loader=None):

    if train_loader is None or valid_loader is None:
        print("Both train loader and validation loader muts be provided")
        exit(1)
    
    valid_loss_min = np.Inf # track chane in validation loss
    start_lr_update = True
    minimum_lr_val_reached = True
    
    initial_lr = optimizer.defaults['lr']
    target_lr = 0.001 # const value.
    start_lr_update_at_epoch = int(nEpochs * lr_update_at_Epoch_perc)        # start updating the learning rate at this percentage of the nEpochs
    minum_Lr_atEpoch_percentage = int(nEpochs * minLr_val_at_Epoch_perc)                # percentage of epochs at which the optimization will reach its minimum learning rate
    lr_update_step = (initial_lr - target_lr) / (minum_Lr_atEpoch_percentage - start_lr_update_at_epoch)
    
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("Training on CUDA!")
        model.cuda()
    else:
        print("CUDA is not available.")
        
    parallel_model = nn.DataParallel(model)     # Encapsulate the model

    for epoch in range(1, nEpochs+1):
        
        if epoch >= start_lr_update_at_epoch and optimizer.defaults['lr'] > 0.001:
            if start_lr_update:
                print("Learning rate starts to be updated towards a value of 0.001")
                start_lr_update = False
                
            optimizer.defaults['lr'] -= lr_update_step
            if optimizer.defaults['lr'] < 0.001:
                optimizer.defaults['lr'] = 0.001 # in case the learning rate update went a bit below 0.001, we reset it to 0.001
                                                 # to avoid an extremly slow optimization                
                if minimum_lr_val_reached:
                    print("Minimum value of learning rate rached (i.e. 0.001)")
                    minimum_lr_val_reached = False

            
        # keep track of training and validation loss
        train_loss = 0
        valid_loss = 0
        train_accuracy = 0
        top3_train_accuracy = 0

        # start counting the elapsed time
        starting_timePoint = time.time()

        ## Training the model ##
        model.train()
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass -> compute predicted outputs by passing inputs to the model
            output = parallel_model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.mean().backward()
            # finally, perform one optimization step (an update of the parameters towards the disminution of error direction)
            optimizer.step()
            # update training loss
            train_loss +=loss.item()*data.size(0)

            # calculating train top-1 accuracy
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # calculating train top-3 accuracy
            npTop3_classes = ps.topk(3, dim=1)[1].cpu().numpy()
            npTarget = target.cpu().numpy()
            top3_train_accuracy += np.mean([1 if npTarget[i] in npTop3_classes[i] else 0 for i in range(0, len(npTarget))])

        # check how much time has elapsed
        time_elapsed = time.time() - starting_timePoint


        validation_accuracy = 0
        top3_validation_accuracy = 0
        ## Validating the model ##
        model.eval()
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # forward pass -> compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item()*data.size(0)

            # calculating validation top-1 accuracy
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # calculating validation top-3 accuracy
            npTop3_classes = ps.topk(3, dim=1)[1].cpu().numpy()
            npTarget = target.cpu().numpy()
            top3_validation_accuracy += np.mean([1 if npTarget[i] in npTop3_classes[i] else 0 for i in range(0, len(npTarget))])

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tIn seconds: {:.4f}'.format(
            epoch, train_loss, valid_loss, time_elapsed))

        # save model if validation loss has decreased!
        if valid_loss <= valid_loss_min:
            print("Validation loss decreased ({:.6f} ---> {:.6f}). Saving model.".format(valid_loss_min,
                                                                                        valid_loss,))
            if bestModelName is None:
                torch.save(model.state_dict(), 'bestModelTrained.pt')
            else:
                torch.save(model.state_dict(), '{}.pt'.format(bestModelName))
                
            valid_loss_min = valid_loss

        if train_stats is not None:
            train_stats = train_stats.append({
                'Epoch' : epoch,
                'Time per epoch' : time_elapsed,
                'Avg time per step' : time_elapsed / len(train_loader.sampler),
                'Train loss' : train_loss,
                'Train accuracy' : train_accuracy / len(train_loader),
                'Train top-3 accuracy' : top3_train_accuracy / len(train_loader),
                'Validation loss' : valid_loss,
                'Validation accuracy' : validation_accuracy / len(valid_loader),
                'Validation top-3 accuracy' : top3_validation_accuracy / len(valid_loader)
            }, ignore_index=True)
        
    # either returns None or returns the train_stats list
    return train_stats