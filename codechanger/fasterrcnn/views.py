from django.shortcuts import render
from rest_framework.decorators import api_view
from .utils import process_data, split_data, data_loader, train_model
import torch
import pickle
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision
import builtins

@api_view(['GET'])
def index(request):
    return render(request, 'index.html')

@api_view(['GET', 'POST'])
def train_model_view(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        num_of_classes = int(request.POST.get('num_of_classes'))
        epochs = int(request.POST.get('epochs'))
        optimizer_choice = request.POST.get('optimizer')
        learning_rate = float(request.POST.get('learning_rate'))
        folder_location = request.POST.get('folder_location')

        df = process_data(folder_location)
        print('data', df.shape)

        train_df, test_df, val_df = split_data(df)
        print('Size of dataset', train_df.shape, test_df.shape, val_df.shape)

        train_dl = data_loader(train_df)
        test_dl = data_loader(test_df)
        val_dl = data_loader(val_df)

        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
                print("Loaded pre-trained model.")
        except FileNotFoundError:
            print("No pre-trained model found, creating a new one.")
            if model_name == "fasterrcnn_resnet50_fpn":
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_of_classes)
            else:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_of_classes)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        print(f"Model is on device: {device}")

        if optimizer_choice == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_choice == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_choice == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        model = train_model(model, train_dl, epochs, optimizer, device)

        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved.")

        return render(request, 'faster.html', {'message': "Model trained successfully!"})

    return render(request, 'faster.html', {'message': "Ready to train the model!"})

