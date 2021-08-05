%   Created by Gabriel Delgado #93661 for FArmer capstone project
%   File: FArmerNN.m
%   Description:
%       This project is for the use of a convolutional neural network for
%       the purposes of identifying lettuces. 
%   History:
%       25-09-2020 - Creation of the first version using standard Matlab
%       practices for neural networks. 

close all;
clear all;

inputSize = [224 224 3]; % size we want to re-size all pictures in dataset to
numClasses = 2; % number of classes for output layer

imds = imageDatastore('C:\Users\gdrpc\Google Drive\University\Capstone\Lechuga', 'IncludeSubfolders', 1, 'LabelSource', 'foldernames'); % load dataset into a datastore object, like an array but not everything in memory automatically
[some, thing] = readimage(imds, 1);
%imds.ReadFcn = @(loc)imresize(imread(loc), inputSize); % Resize to target, in this case we made everything as large as the samllest image in the dataset.
imds.ReadFcn = @(loc)imresize(imread(loc), [224 224]); 

% show some of the images of the dataset
% perm = randperm(30,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

labelCount = countEachLabel(imds); % count how many files we have in each class for training and validation

numTrainFiles = 20; % Specify how many files from each class to use for training

[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize'); % assign files to training and validation vectors

% Define a Convolutional Neural Network's architecture
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];

% Specify the training options, number of epochs, plots, etc...
options = trainingOptions('sgdm', ...
    'MaxEpochs',20, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',15, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain, layers, options); % Begin training the network with the specified training data and architecture

YPred = classify(net, imdsValidation); % Test the network with new data and observe its accuracy.
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

analyzeNetwork(net); % This command allows you to see the architecture of the network with GUI

%Show results
figure('Name', 'Results');
for i = 1 : 50
    subplot(10, 5, i);
    imshow(imds.Files{i});
    title(YPred(i, 1));
end;

TestNet = net;
save TestNet_224; % Rename and save trained network

%load TestNet;
% imdsFinalTest = imageDatastore('C:\Users\gdrpc\Google Drive\University\Capstone\New folder', 'IncludeSubfolders', 1, 'LabelSource', 'foldernames');
% imdsFinalTest.ReadFcn = @(loc)imresize(imread(loc), [150 150]); 
% TestResults = classify(TestNet, imdsFinalTest);

%Show results
% figure('Name', 'Test Results');
% for i = 1 : 50
%     subplot(10, 5, i);
%     imshow(imdsFinalTest.Files{i});
%     title(TestResults(i, 1));
% end;

%Export Net in ONNX format
exportONNXNetwork(net, 'Lettuce_Net_224.onnx');
