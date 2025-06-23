import argparse
import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path) # Load CSV into pandas
    return df


def preprocess(df):
    df_numeric = df.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric)
    return X, scaler


def train_autoencoder(X_train,hidden_neurons=[64, 32, 32, 64],epochs=50,batch_size=32,contamination=0.01): 
    ae = AutoEncoder(hidden_neurons=hidden_neurons,epochs=epochs,batch_size=batch_size,contamination=contamination,verbose=1,random_state=42) # Define the autoencoder
    ae.fit(X_train) #train
    return ae


def detect_anomalies(ae, X):
    scores = ae.decision_function(X) # get anonaly scores
    labels = ae.labels_
    return scores, labels


def plot_reconstruction_error(scores, output_path):
    # Define output format and names
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50)
    plt.title('Reconstruction Error Dist')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()


def main():
    ## Parse arguments
    parser = argparse.ArgumentParser(description='Fraud Detection using AutoEncoder (PyOD)')
    parser.add_argument('--data', # thios would be the datasource file locationtype=str,required=True,help='Path to the transactions CSV file'
                        )
    parser.add_argument('--output_dir', # This is where to save the outputtype=str,default='output',help='Directory to save logs, results, and plots'
                        )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True) # Create output directory if it doesn't exist

    logging.basicConfig(filename=os.path.join(args.output_dir, 'run.log'),level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

    ## Adding logging to output for better debugging
    logging.info('Loading data from %s', args.data)
    df = load_data(args.data)

    logging.info('Preprocessing data')
    X, scaler = preprocess(df)

    logging.info('Splitting data into train and test sets')
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    logging.info('Training AutoEncoder model')
    ae = train_autoencoder(X_train)

    logging.info('Detecting anomalies on test set')
    scores, labels = detect_anomalies(ae, X_test)

    df_results = pd.DataFrame({ # Saving results to csv file'reconstruction_error': scores,'anomaly_label': labels
        })

    results_csv = os.path.join(args.output_dir, 'results.csv')
    df_results.to_csv(results_csv, index=False)
    logging.info('Results saved to %s', results_csv)

    plot_path = os.path.join(args.output_dir, 'error_distribution.png') # plot histogram output
    plot_reconstruction_error(scores, plot_path)
    logging.info('Reconstruction error plot saved to %s', plot_path)


if __name__ == '__main__':
    main()