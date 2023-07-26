import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp

#############################################################################
################        DATASET & TRANSFORMERS       ########################
#############################################################################


# Paths to Data and Masks
data_path = "./DataSets/DataSet_640x352"
masks_path = "./Masks/MasksSet_640x352"

# Transformations
transformation = transforms.Compose([
    # transform to tensors
    transforms.ToTensor(),
    # Garyscale the pixels.
    transforms.Grayscale(num_output_channels=3)
])

transformationMasks = transforms.Compose([
    # transform to tensors
    transforms.ToTensor(),
    # Garyscale the pixels.
    transforms.Grayscale(num_output_channels=1)
])

# Load all of the images and transforming them
full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transformation)  # 107

# 15 img for test
n_test = int(15/107 * len(full_dataset))  # 15
test_set = torch.utils.data.Subset(full_dataset, range(n_test))  # 15

train_validation_set = torch.utils.data.Subset(full_dataset, range(n_test, len(full_dataset)))  # 92

# 15 img for validation
n_val = int(15/(107-15) * len(train_validation_set))  # 15
validation_set = torch.utils.data.Subset(train_validation_set, range(n_val))  # 15

train_set = torch.utils.data.Subset(train_validation_set, range(n_test, len(train_validation_set)))  # 77

# Load all of the masks and transforming them
full_dataset_masks = torchvision.datasets.ImageFolder(root=masks_path, transform=transformationMasks)  # 107

# 15 img for test
n_test = int(15/107 * len(full_dataset_masks))  # 15
test_set_masks = torch.utils.data.Subset(full_dataset_masks, range(n_test))  # 15

train_validation_set_masks = torch.utils.data.Subset(full_dataset_masks, range(n_test, len(full_dataset_masks)))  # 92

# 15 img for validation
n_val = int(15/(107-15) * len(train_validation_set_masks))  # 15
validation_set_masks = torch.utils.data.Subset(train_validation_set_masks, range(n_val))  # 15

train_set_masks = torch.utils.data.Subset(train_validation_set_masks, range(n_test, len(train_validation_set_masks)))  # 77

#############################################################################
######################        INDEX ERROR       #############################
#############################################################################


# Calculate TP, FP, TN, FN
def get_stats(output, target, threshold):
    
    batch_size, num_classes, *dims = target.shape

    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    output = output.detach().numpy()
    target = target.detach().numpy()

    # Sharpe the different between the pixcels by threshold
    output = np.where(output >= threshold, 1, 0)
    target = np.where(target >= threshold, 1, 0)

    # Calculate TP, FP, TN, FN for batch retrun array of the result per Image
    tp = np.zeros((batch_size, 1))
    fp = np.zeros((batch_size, 1))
    fn = np.zeros((batch_size, 1))
    tn = np.zeros((batch_size, 1))

    for batch in range(batch_size):
        # TP = all the pixcels there are 1 in both pictures
        tp_temp = np.sum(output[batch][0] * target[batch][0])
        # FP = all the pixcels there are 1 in target but zero in the output
        fp[batch][0] = np.sum(target[batch][0]) - tp_temp
        # FN = all the pixcels there are 1 output but zero in the target
        fn[batch][0] = np.sum(output[batch][0]) - tp_temp
        # TN = all the pixcels there are 0 in both pictures
        tn[batch][0] = (dims[0] * dims[1]) - (tp_temp + fp[batch][0] + fn[batch][0])
        tp[batch][0] = tp_temp

    return tp, fp, fn, tn

# Return the accuracy batch of images
def accuracy_calc(tp, fp, fn, tn):
    return np.mean((tp + tn) / (tp + fp + fn + tn))

# Return the accuracy batch of images (Stiffer)
def iou_calc(tp, fp, fn):
    return np.mean(tp / (tp + fp + fn))

#############################################################################
######################          TRAIN           #############################
#############################################################################


def train(model, train_loader, train_loader_masks, optimizer, loss_criteria, num_epochs=10, verbose=False):

    model.train()

    # Sets all the required parameters that need to return from this train
    losses = []
    epochs_loss = []
    epoch_list = []
    accuracy = []

    # Epochs
    for epoch in range(num_epochs):

        total_loss = 0
        num_of_batchs = 0

        if verbose:
            print("epoch: ", epoch + 1)

        acc = 0

        # Load Mask and Data
        for data, masks in zip(train_loader, train_loader_masks):

            num_of_batchs += 1
            img, _ = data
            mask, _ = masks
            # Train on Data to find sigmantition
            reconstructed_input = model(img)
            # Calculate the loss
            loss = loss_criteria(reconstructed_input, mask)
            # Set the optimaizer to zero for each batch
            optimizer.zero_grad()
            # Preform backpropogation
            loss.backward()
            # Update the wights matrix
            optimizer.step()
            # Save Loss, Accuraty for this batch
            losses.append(loss.item())
            total_loss += loss.item()
            # Index Error
            tp, fp, fn, tn = get_stats(reconstructed_input, mask, 0.5)
            acc += iou_calc(tp, fp, fn)

            if verbose:
                print("Batch: ", num_of_batchs, " Average Batch Loss: ", loss.item())

        total_loss /= num_of_batchs

        if verbose:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, float(total_loss)))
            print("Train accuracy:", acc / num_of_batchs)

        epochs_loss.append(total_loss)
        accuracy.append(acc / num_of_batchs)
        epoch_list.append(epoch + 1)

    return epochs_loss, accuracy, epoch_list

#############################################################################
#######################          TEST           #############################
#############################################################################


def test(model, test_loader, test_loader_masks, loss_criteria, title, verbose):

    model.eval()

    # Sets all the required parameters that need to return from this test
    outputs = []
    total_loss = 0
    iou_score = 0
    accuracy = 0
    scroll_success_rate = 0
    background_success_rate = 0

    with torch.no_grad():

        print(title, '\n')

        for data, masks in zip(test_loader, test_loader_masks):

            img, _ = data
            mask, _ = masks
            
            # Test on Data to find sigmantition
            predict = model(img)
            
            # Calculate the loss
            loss = loss_criteria(predict, mask)
            total_loss += loss.item()
            
            outputs.append([img, mask, predict])

            # Index Error  
            tp, fp, fn, tn = get_stats(predict, mask, 0.5)
            accuracy += accuracy_calc(tp, fp, fn, tn)
            iou_score += iou_calc(tp, fp,fn)
            scroll_success_rate += np.mean(tp / (tp + fp))
            background_success_rate += np.mean(tn / (tn + fn))

        total_loss /= len(test_loader.dataset)
        accuracy /= len(test_loader.dataset)
        iou_score /= len(test_loader.dataset)

        if verbose:
            print('Average loss: {:.6f}'.format(total_loss))
            print('Accuracy: {:.6f}'.format(accuracy))
            print('iou score: {:.6f}\n'.format(iou_score))

        return outputs, total_loss, iou_score, scroll_success_rate, background_success_rate

#############################################################################
#######################          MAIN           #############################
#############################################################################


if __name__ == "__main__":

    # Initalatize the model
    torch.manual_seed(42)

    results = []

    # Set Hyper Parameters to check
    backbones = ['resnet34', 'timm-regenty_320']
    encoder_depths = [3, 4, 5]
    decoder_channels = [[64, 32, 16], [128, 64, 32, 16], [256, 128, 64, 32, 16]]
    num_of_epcohs = [5, 20, 50]
    batch_sizes = [10]
    loss_criterias = [smp.losses.DiceLoss(mode='binary'), smp.losses.TverskyLoss(mode='binary')]
    num_of_optimizers = 5
    batch_size_test = 1

    # Set the best Hyper parameters
    best_backbone = ''
    best_encoder_depth = -1
    best_decoder_channel = []
    best_batch_size = -1
    best_acc = -1
    best_num_of_epochs = 50
    best_loss_criteria = smp.losses.DiceLoss(mode='binary')
    best_optimizer = -1

    # Set the validation loder
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_test, shuffle=False)
    validation_loader_masks = torch.utils.data.DataLoader(validation_set_masks, batch_size=batch_size_test, shuffle=False)

    # Serach for the best Hyper parameters
    for backbone in backbones:
        for encoder_depth, decoder_channel in zip(encoder_depths, decoder_channels):
            for epoch in num_of_epcohs:
                for batch_size_train in batch_sizes:
                    for optimizer in range(num_of_optimizers):
                        for loss_criteria in loss_criterias:

                            # Initalatize the model with different parameters
                            model = smp.UnetPlusPlus(encoder_name = backbone, encoder_depth = encoder_depth, decoder_channels = decoder_channel,
                                                    in_channels = 3, classes = 1, activation = "sigmoid", decoder_use_batchnorm = True)

                            # Initalatize the optimizers with different parameters
                            optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4),
                                          torch.optim.Adam(model.parameters(), lr=1e-3),
                                          torch.optim.Adamax(model.parameters(), lr=1e-4),
                                          torch.optim.Adadelta(model.parameters(), lr=1e-4),
                                          torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)]

                            print("\nSerach for the best Hyper parameters with ==>\n")
                            print("Backbone:", backbone)
                            print("Encoder depth:", encoder_depth)
                            print("Decoder channel:", decoder_channel)
                            print("Epochs:", epoch)
                            print("Batch size", batch_size_train)
                            print("Loss criteria", loss_criteria)
                            print("Optimizer", optimizers[optimizer])

                            # Set the train loder with different parameters
                            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False)
                            train_loader_masks = torch.utils.data.DataLoader(train_set_masks, batch_size=batch_size_train, shuffle=False)

                            # Train the model on Train
                            epochs_loss, acc, epoch_list = train(model, train_loader=train_loader, train_loader_masks=train_loader_masks,
                                                                optimizer=optimizers[optimizer], loss_criteria=loss_criteria, num_epochs=epoch, verbose=True)

                            # Test the model on Validation
                            val_output, val_loss, acc_val, scrolls, background = test(model, test_loader=validation_loader, test_loader_masks=validation_loader_masks,
                                                                loss_criteria=loss_criteria, title="Validation", verbose=True)

                            # If this is the best model until now?
                            if acc_val > best_acc:
                                best_acc = acc_val
                                best_backbone = backbone
                                best_encoder_depth = encoder_depth
                                best_decoder_channel = decoder_channel
                                best_batch_size = batch_size_train
                                best_loss_criteria = loss_criteria
                                best_optimizer = optimizer
                            
                            results.append([acc_val, backbone, encoder_depth, decoder_channel, epoch, batch_size_train, loss_criteria, optimizer])

                            print("Best accuracy until now:", best_acc)
                            print("Best backbone until now:", best_backbone)
                            print("Best encoder depth until now:",best_encoder_depth)
                            print("Best decoder channel until now:", best_decoder_channel)
                            print("Best optimizer until now:",optimizers[best_optimizer])
                            print("Best loss criteria until now:", best_loss_criteria)
                            print("Best batch size for train until now:", best_batch_size)

    # Set the train + validation and test loder 
    train_validation_loader = torch.utils.data.DataLoader(train_validation_set, batch_size=best_batch_size, shuffle=False)
    train_validation_loader_masks = torch.utils.data.DataLoader(train_validation_set_masks, batch_size=best_batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
    test_loader_masks = torch.utils.data.DataLoader(test_set_masks, batch_size=batch_size_test, shuffle=False)

    # Initalatize the model with the best hyper parameters
    model = smp.UnetPlusPlus(encoder_name = best_backbone, encoder_depth = best_encoder_depth, decoder_channels = best_decoder_channel,
                            in_channels = 3, classes = 1, activation = "sigmoid", decoder_use_batchnorm = True)

    # Initalatize the best optimizer with the best hyper parameters
    best_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4),
                       torch.optim.Adam(model.parameters(), lr=1e-3),
                       torch.optim.Adamax(model.parameters(), lr=1e-4),
                       torch.optim.Adadelta(model.parameters(), lr=1e-4),
                       torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)]

    # Train on Train + Validation
    train_losses, acc_train, epoch_list = train(model, train_loader=train_validation_loader, train_loader_masks=train_validation_loader_masks,
                                                optimizer=best_optimizers[best_optimizer], loss_criteria=best_loss_criteria, num_epochs=best_num_of_epochs, verbose=True)

    # Test the model on Test
    test_output, test_loss, acc_test, scroll_success_rate, background_success_rate = test(model, test_loader=test_loader, test_loader_masks=test_loader_masks,
                                            loss_criteria=best_loss_criteria, title="Test", verbose=True)

    print("best accuracy:", acc_test)
    print("best number of epochs:", best_num_of_epochs)
    print("best optimizer:", best_optimizers[best_optimizer])
    print("best loss criteria:", best_loss_criteria)
    print("best batch size for train:", best_batch_size)

    # Plot Train Loss Graph
    plt.title('Train Loss Graph')
    plt.plot(epoch_list, train_losses)
    plt.show()

    # Plot Train Accuracy Graph
    plt.title('Train Accuracy Graph')
    plt.plot(epoch_list, acc_train)
    plt.show()

    print("Accuracy on test: ", acc_test)
    print("Scroll success rate: ", scroll_success_rate)
    print("Background success rate: ", background_success_rate)

    # Plot 5 results of the Module
    for i in range(len(test_output)):

        plt.subplot(1, 4, 1), plt.axis('off'), plt.imshow(test_output[i][0].detach().numpy()[0, 0, :, :].T, cmap='gray')
        plt.subplot(1, 4, 2), plt.axis('off'), plt.imshow(test_output[i][1].detach().numpy()[0, 0, :, :].T, cmap='gray')
        plt.subplot(1, 4, 3), plt.axis('off'), plt.imshow(test_output[i][2].detach().numpy()[0, 0, :, :].T, cmap='gray')
        plt.show()
    
    # List all the Hyper Parameters were tested
    index = 0
    print("\n \n The Hyper Parameters were tested and their results")
    for result in results:
        index += 1
        print("Model No. ", index)
        print("Score: ", result[0])
        print("Backbone: ", result[1])
        print("Encoder depth: ", result[2])
        print("Decoder depth: ", result[3])
        print("No. of epochs: ", result[4])
        print("Batch size: ", result[5])
        print("Loss Criteria: ", result[6])
        print("Optimizer: ", best_optimizers[result[7]], '\n')
