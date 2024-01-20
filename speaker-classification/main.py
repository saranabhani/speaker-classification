import os
import argparse
import numpy as np
from preprocessor import mel_spectrogram
from models.ShallowNet import ShallowNet
from models.AlexNet import AlexNet
from models.VGGNet import VGGNet
from models.CnnRnn import CnnRnn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import SGD, Adam
from contextlib import redirect_stdout

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=False, default="vgg", help="model architecture to use vgg, alexnet, shallow")
ap.add_argument("-l", "--loss", required=False, default="categorical_crossentropy", help="loss function")
ap.add_argument("-o", "--optimizer", required=False, default="sgd", help="optimizer to use Adam or SGD")
ap.add_argument("-e", "--epochs", required=False, default=50, help="number of epochs", type=int)
ap.add_argument("-b", "--batch_size", required=False, default=32, help="batch size", type=int)
ap.add_argument("-eb", "--e_batch_size", required=False, default=10, help="evaluation batch size", type=int)
ap.add_argument("-met", "--metric", required=False, default="accuracy", help="metric to use")
ap.add_argument("-lr", "--learning_rate", required=False, default=0.005, help="learning rate", type=float)
ap.add_argument("-do", "--dropout", required=False, default=0.5, help="dropout rate", type=float)
ap.add_argument("-fs", "--frame_size", required=False, default=1024, help="frame size", type=int)
ap.add_argument("-nm", "--n_mels", required=False, default=128, help="number of mels", type=int)
ap.add_argument("-cs", "--chunk_seconds", required=False, default=3, help="chunk seconds", type=int)
ap.add_argument("-mn", "--model_name", required=True, default="model", help="model name")
args = vars(ap.parse_args())


# main function
if __name__ == '__main__':
    # load the dataset from disk
    print("[INFO] loading audio files...")
    dataset_path = args["dataset"]
    speaker_wav = dict()
    for speaker_dir in os.listdir(dataset_path):
        if speaker_dir.startswith('.'):
            continue
        for file in os.listdir(os.path.join(dataset_path, speaker_dir)):
            wav_file_path = os.path.join(dataset_path, speaker_dir, file)
            if speaker_dir not in speaker_wav.keys():
                speaker_wav[speaker_dir] = [spec for spec in mel_spectrogram(wav_file_path, args["frame_size"],
                                                                             args["n_mels"],
                                                                             chunk_seconds=args["chunk_seconds"])]
            else:
                speaker_wav[speaker_dir].extend(mel_spectrogram(wav_file_path, chunk_seconds=args["chunk_seconds"]))
    speaker_wav_expanded = [(id, spect) for id, spects in speaker_wav.items() for spect in spects]
    data_df = pd.DataFrame(speaker_wav_expanded, columns=['id', 'feature'])

    labels_encoder = LabelEncoder().fit(data_df['id'])
    labels_encoded = labels_encoder.transform(data_df['id'])
    labels = to_categorical(labels_encoded)

    x_train = np.array([np.array(val) for val in data_df['feature']])
    x_train, x_test, y_train, y_test = train_test_split(x_train, labels, test_size=0.2, stratify=labels, random_state=7)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=7)
    print(f"X_train shape: {x_train.shape}")
    print(f"val shape: {x_val.shape}")
    print(f"test shape: {x_test.shape}")
    print("[INFO] compiling model...")
    if args["model"] == "alexnet":
        model = AlexNet()
    elif args["model"] == "vgg":
        model = VGGNet()
    elif args["model"] == "shallow":
        model = ShallowNet()
    elif args["model"] == "mixed":
        model = CnnRnn()
    else:
        raise Exception("Model not supported")
    opt = SGD(learning_rate=args["learning_rate"]) if args["optimizer"] == "sgd" else Adam(learning_rate=args["learning_rate"])
    model = model.build(width=x_train.shape[2], height=x_train.shape[1], depth=1, classes=len(labels_encoder.classes_), dropout=args["dropout"])
    model.compile(loss=args["loss"], optimizer=opt, metrics=[args["metric"]])
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=10,        # Number of epochs with no improvement after which training will be stopped
                               min_delta=0.001,    # Minimum change to qualify as an improvement
                               mode='min',         # The training will stop when the quantity monitored has stopped decreasing
                               verbose=1)
    # save model summary
    model_dir = f'./{args["model_name"]}'
    os.mkdir(model_dir)
    with open(os.path.join(model_dir, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # train the network
    print("[INFO] training network...")
    H = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=args["batch_size"], epochs=args["epochs"],
                  verbose=1, callbacks=[early_stopping])

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(x_test, batch_size=args["e_batch_size"])
    cls_report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labels_encoder.classes_)

    # Save the report to a text file
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(cls_report)
    print(cls_report)

    early_epoch = early_stopping.stopped_epoch + 1 if early_stopping.stopped_epoch > 0 else args["epochs"]
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, early_epoch), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, early_epoch), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, early_epoch), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, early_epoch), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'eval_plot.png'))
    
    cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    with open(os.path.join(model_dir, 'confusion_matrix.txt'), 'w') as f:
        for row in cm:
            f.write(' '.join([str(a) for a in row]) + '\n')
    
    # Calculating accuracy for each label
    accuracies = np.diag(cm) / np.sum(cm, axis=1)
    
    # Creating a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=[f'True_{i}' for i in labels_encoder.classes_],
                         columns=[f'Pred_{i}' for i in labels_encoder.classes_])
    # Adding accuracies as a new column
    cm_df['Accuracy'] = accuracies
    # Saving the DataFrame to a CSV file
    cm_df.to_csv(os.path.join(model_dir,'confusion_matrix_with_accuracy.csv'), index=True)



    
    

