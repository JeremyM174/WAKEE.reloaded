import os, shutil, glob, boto3, time, random, requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from botocore.exceptions import ClientError
import onnx
import onnxruntime as ort
from PIL import Image
import torchvision.transforms.v2 as transforms
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report 
from tqdm import tqdm 
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import git
from git import Repo



# =====DAILY EXTRACT/LOAD=====

def daily_extraction(**context):
    csv_list = []
    frame_list = []

    for csv_file in os.listdir("src/temp"):
        if csv_file.lower().endswith(".csv"):
            csv_list.append(csv_file)

    for frame in csv_list:
        frame = frame.replace(".csv", "")
        frame_list.append(frame)

    print(csv_list)
    print(frame_list)

    rated_files = glob.glob(os.path.join("src/temp", "*.csv"))
    df_rated = pd.concat((pd.read_csv(f) for f in rated_files), ignore_index=True)
    df_rated.to_csv(f"src/temp/daily_ratings{date.today()}.csv", index=False)
    print("CSV successfully created!")
    context['task_instance'].xcom_push(key='frame_list', value=frame_list)


def daily_load(**context):
    frame_list = context['task_instance'].xcom_pull(key='frame_list')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    session = boto3.Session(
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    s3 = session.resource("s3")
    bucket = s3.Bucket("deploymentproject174")

    bucket.upload_file(f"src/temp/daily_ratings{date.today()}.csv", f"wakee_reloaded/daily_ratings{date.today()}.csv")
    print("CSV file pushed to S3!")
    for frame_file in frame_list:
        bucket.upload_file(f"src/temp/{frame_file}", f"wakee_reloaded/{frame_file}")
    print("Frames pushed to S3!")


def daily_clean():
    shutil.rmtree("src/temp", ignore_errors=True)
    print("Full temp directory deleted!")
    os.mkdir("src/temp")
    print("Temp directory created!")



# =====WEEKLY TRAINING=====



# ===PREPARE DATA===

def weekly_s3_extract():
    s3 = boto3.client("s3")
    bucket_name=os.getenv("S3_BUCKET_NAME")
    s3_prefix=os.getenv("S3_WORK_FOLDER")
    local_dir="src/training/archive"
    paginator = s3.get_paginator("list_objects_v2")

    # pull data from S3
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            s3_key = obj["Key"]

            # skip "directory markers" (zero-byte objects ending with /)
            # if s3_key.endswith("/"):
            #     continue

            # compute local file path
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_dir, relative_path)

            # make sure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            try:
                s3.download_file(bucket_name, s3_key, local_path)
                print(f"Downloaded {s3_key} into {local_path}")
            except Exception as e:
                print(f"Failed to download {s3_key}: {e}")
                continue


def organize_data():
    local_dir="src/training/archive"

    # move frames into their working directory (matching training paths)
    for file in os.listdir(local_dir):
        if file.lower().endswith(".jpg"):
            shutil.move(f"{local_dir}/{file}", f"{local_dir}/DataSet/{file}")
    print("Frames moved!")

    # concatenate ratings (csv > dataframe > csv)
    rfiles = glob.glob(os.path.join(local_dir, "*.csv"))
    df_concat = pd.concat((pd.read_csv(f) for f in rfiles), ignore_index=True)
    df_concat.head()
    df_concat.to_csv(f'{local_dir}/labels_daisee_continous.csv', index=False)
    print("Training csv written over!")

    files_removal = glob.glob(os.path.join(local_dir, "daily_ratings*.csv"))
    for i in files_removal:
        os.remove(i)
    print("Folder cleaned!")


# ===CHECK MLFLOW===

def check_mlflow(**context):
    mlflow.set_tracking_uri(os.getenv("TRACKING_SERVER_URI"))
    client = MlflowClient()
    prod_versions = client.get_latest_versions(name="reloaded_model", stages=["Production"])

    prod_model_version = prod_versions[0]
    prod_run_id = prod_model_version.run_id
    prod_run = client.get_run(prod_run_id)
    prod_mae = prod_run.data.metrics["MAE"]

    print(f"Current best MAE: {prod_mae}")
    context['task_instance'].xcom_push(key='prod_mae', value=prod_mae)


# ===TRAINING===

def new_training(**context):
    prod_mae = context['task_instance'].xcom_pull(key='prod_mae')

    class EmotionDataset(Dataset):
        def __init__(self, df, img_dir, emotion_cols, transform=None):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform
            self.emotion_cols = emotion_cols

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['image_path'])

            emotion_labels = row[self.emotion_cols].values.astype(float)
            emotion_labels = torch.tensor(emotion_labels, dtype=torch.float32)

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, emotion_labels
    


    DATA_DIR = "src/training/archive"
    CSV_FILE = os.path.join(DATA_DIR, 'labels_daisee_continous.csv')
    IMAGE_DIR = os.path.join(DATA_DIR, 'DataSet')

    # Define your emotions here, in the same order as your columns in the CSV
    EMOTION_LABELS = ['boredom', 'confusion', 'engagement', 'frustration']
    NUM_CLASSES = len(EMOTION_LABELS)

    BATCH_SIZE = 32 
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    FREEZE_FEATURES = True # True to fine-tune only the last layer, False to fine-tune the entire model

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    EXPERIMENT_NAME="WAKEE.reloaded"
    mlflow.set_tracking_uri(os.environ["TRACKING_SERVER_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    mlflow.pytorch.autolog()



    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    full_df = pd.read_csv(CSV_FILE)

    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_dataset = EmotionDataset(df=train_df, img_dir=IMAGE_DIR, emotion_cols=EMOTION_LABELS, transform=data_transforms['val'])
    val_dataset = EmotionDataset(df=val_df, img_dir=IMAGE_DIR, emotion_cols=EMOTION_LABELS, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print(f"Training dataset size : {dataset_sizes['train']}")
    print(f"Testing dataset size : {dataset_sizes['val']}")



    model_ft = models.efficientnet_b4(weights="IMAGENET1K_V1")

    # layers freeze
    for param in model_ft.parameters():
        param.requires_grad = False

    # fine_tune_at = len(list(model_ft.children())) - 10
    for name, param in list(model_ft.named_parameters())[-17:]:
        print(name)
        param.requires_grad = True

    print("Définition de la dernière couche...")

    # replacing the final fully-connected layer
    num_ftrs = model_ft.classifier[1].in_features

    model_ft.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )

    # sending the model to the correct device (GPU or CPU)
    model_ft = model_ft.to(DEVICE)



    summary(model_ft, input_size=(1, 3, 224, 224))



    epochs=random.randint(2,5)
    criterion_emotions = nn.MSELoss()
    # criterion_emotions = weighted_mse_loss => to try

    # Only parameters that require gradients will be optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler (reduces LR after a certain number of epochs)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



    def train_model(model, 
                    criterion_emotions, 
                    optimizer, 
                    train_loader, 
                    val_loader, 
                    device, 
                    scheduler=None,
                    epochs=NUM_EPOCHS,
                    output_names=EMOTION_LABELS):
        
        since = time.time()
        
        history = {'train_loss': [],
                    'val_loss': [],
                    'val_MAE': [],
                    'val_MSE': [],
                    'val_RMSE': [],
                    'val_R2': []}

        model.to(device)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print('-' * 30)

            ### -------- TRAIN --------
            model.train()
            running_loss = 0.0


            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion_emotions(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item() * inputs.size(0)
                

            epoch_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)

            ### -------- VAL --------
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_true_values = []
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion_emotions(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # Convert tensors to numpy to calculate sklearn metrics
                    val_predictions.extend(outputs.cpu().numpy())
                    val_true_values.extend(labels.cpu().numpy())

                    preds = torch.round(outputs)
                    preds = torch.clip(preds, 0, 3)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            # Concatenation of all batches
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Convert lists to numpy array for sklearn
            val_true_values_np = np.array(val_true_values)
            val_predictions_np = np.array(val_predictions)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_MAE = mean_absolute_error(val_true_values_np, val_predictions_np)
            val_epoch_MSE = mean_squared_error(val_true_values_np, val_predictions_np)
            val_epoch_RMSE = np.sqrt(mean_squared_error(val_true_values_np, val_predictions_np))
            val_epoch_r2 = r2_score(val_true_values_np, val_predictions_np)
            
            history['val_loss'].append(val_epoch_loss)
            history['val_MAE'].append(val_epoch_MAE)
            history['val_MSE'].append(val_epoch_MSE)
            history['val_RMSE'].append(val_epoch_RMSE)
            history['val_R2'].append(val_epoch_r2)
        
            print(f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {val_epoch_loss:.4f} | "
                f"Val MAE: {val_epoch_MAE:.4f} | "
                f"Val MSE: {val_epoch_MSE:.4f} | "
                f"Val RMSE: {val_epoch_RMSE:.4f} | "
                f"Val R2: {val_epoch_r2:.4f} \n ")
            
            # Displaying the multi-label classification report
            # print("\n📊 Rapport de classification multilabels-multiouputs :")
            
            # target_names_classes = ['0', '1', '2', '3'] # Class names for the report

            # for i, names in enumerate(output_names):
            #     print(f"\n--- Variable {names} ---")
                
            #     # The real classes for the variable i
            #     y_true_var_i = all_labels[:, i]
                
            #     # The classified predictions for variable i
            #     y_pred_var_i = all_preds[:, i]
                
            #     # Check that the values are integers and within the expected range
            #     if not np.all(np.isin(y_true_var_i, [0, 1, 2, 3])):
            #         print(f"Attention: y_true pour la variable {names} contient des valeurs hors de [0, 3] ou non entières après arrondi.")
            #     if not np.all(np.isin(y_pred_var_i, [0, 1, 2, 3])):
            #         print(f"Attention: y_pred pour la variable {names} contient des valeurs hors de [0, 3] ou non entières après arrondi.")

            #     # Generate and display the report.
            #     print(classification_report(y_true_var_i, y_pred_var_i, target_names=target_names_classes, zero_division=0))
            

            if scheduler:
                scheduler.step()

        time_elapsed = time.time() - since
        print(f"\nTraining duration: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        return history



    # training start
    with mlflow.start_run(experiment_id = experiment.experiment_id):
        history = train_model(model=model_ft,
                                criterion_emotions=criterion_emotions,
                                optimizer=optimizer_ft,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=DEVICE,
                                scheduler=exp_lr_scheduler,
                                epochs=2, #epochs, # replace with 2 for quick training
                                output_names=EMOTION_LABELS)

        mlflow.log_metrics({
            "MAE":history["val_MAE"][-1],
            "MSE":history["val_MSE"][-1],
            "RMSE":history["val_RMSE"][-1],
            "R²":history["val_R2"][-1],
            "Val train":history["train_loss"][-1],
            "Val loss":history["val_loss"][-1]
        })
        
        mlflow.log_params({
            "Epochs":epochs,
            "Loss function":criterion_emotions,
            "Optimizer":optimizer_ft,
            "LR/step size":exp_lr_scheduler.step_size,
            "LR/gamma":exp_lr_scheduler.gamma
        })
        current_mae = history["val_MAE"][-1]
        print(f"Current training MAE: {current_mae}")

        if current_mae < prod_mae:
            print(f"\nNew best MAE: {current_mae} against former {prod_mae}! Logging model...")
            mlflow.set_tracking_uri(os.getenv("TRACKING_SERVER_URI"))
            client = MlflowClient()

            example = np.random.randn(1, 3, 224, 224).astype(np.float32)
            signature = infer_signature(example, model_ft(torch.from_numpy(example)).detach().numpy())
            model_info = mlflow.pytorch.log_model(model_ft, name="model", input_example=example, signature=signature)

            registered_model = mlflow.register_model(model_uri=model_info.model_uri, name="reloaded_model")
            client.transition_model_version_stage(
                name="reloaded_model",
                version=registered_model.version,
                stage="Production"
            )

            print("\n...done. Retrieving model locally...")
            # Dissociating local retrieval: predict API expects a .onnx file as model
            onnx_export_path = "src/WAKEE_reloaded_API/model/daisee_model.onnx"
            dummy_input = torch.randn(1, 3, 224, 224)

            torch.onnx.export(
                model_ft,                  
                dummy_input,               
                onnx_export_path,          
                input_names=['input'],     
                output_names=['output'],   
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=11,
                dynamo=False,
                do_constant_folding=True   # Optimization for constants
            )
            print(f"\n...done! Made available at: {onnx_export_path}")

            print("\nUpdating API model...")
            repo = Repo("src/WAKEE_reloaded_API")
            repo.git.add("model/daisee_model.onnx")
            repo.git.commit(m="API model update")
            repo.git.push("origin", "main")
            print("\nModel pushed, deployment will restart soon!")

        
        else:
            print(f"\nPerformance target missed: current MAE {current_mae} higher than reference {prod_mae}!")