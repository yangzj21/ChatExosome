from flask import Flask, request
import requests
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.XFFT import FFT
from utils.spectra_dataset import *

def build_cell_exosome():
    data_shape = (1, 0, 1024)
    dim_rate = 6 if data_shape[0] == 1 else 2
    cfg = {
        "data_shape": (1, 0, 1024),
        "num_classes": 2,
        "patch_size": 8,
        "embed_dim": 8 * data_shape[0] * dim_rate,  
        "depths": (2, 2, 6, 2),  
        "num_heads": tuple(i * data_shape[0] * dim_rate for i in (1, 2, 4, 8)),  
        "down_sample_rate": None,  
        "checkpoint": './checkpoints/pretrained_model.pth'
    }

    def datasetLoader(X, y):
        generators = [identity_transform, fd_transform, sd_transform, range_scaling_transform, snv_transform, 
              l2norm_transform, smooth_transform, fourier_transform, denoise_by_wavelets]
        transform = Interp1dResize(data_shape[-1])
        return SpectraDataset(X, y, output_shape=data_shape, transform=transform, generators=generators)

    # model
    model = FFT(**cfg)
    

    return {
        'model': model,
        'dataset_loader': datasetLoader,
        'class_values_map': {
            0: 'Normal',
            1: 'Cancer'
        },
        'similar_X_df': pd.read_csv('./data/cell_df.csv', index_col=0).iloc[:, :-1],
    }

models = {
    'cell_exosome': build_cell_exosome(),
}

def cosine_similarity(df1, df2=None):
    """
        Calculate the cosine similarity between each row in the dataframe df and df2.
    """
    # cos = x*y/|x|*|y|
    dfnorm = df1.apply(np.linalg.norm, axis=1)
    if df2 is None:
        df2 = df1
        df2norm = dfnorm
    else:
        df2norm = df2.apply(np.linalg.norm, axis=1)

    cosine_sim = df1.dot(df2.T).divide(np.outer(dfnorm, df2norm))
    sim_df = pd.DataFrame(cosine_sim.values, index=df1.index, columns=df2.index)

    return sim_df

def is_similarity(df, testDf, threshold=0.7, ratio=0.75):
    '''
        cosine similarity > threshold,ratio > ratio
    '''
    sim_df = cosine_similarity(df, testDf)
    th_df = (sim_df > threshold).sum()
    return th_df >= (len(df)*ratio)


def try_parse_data(fileOrURL: str):
    if fileOrURL.startswith('http'):
        data_resp = requests.get(fileOrURL)
        if data_resp.status_code != 200:
            raise Exception('http get failed')
        
        fileOrURL = '/tmp/data'
        with open(fileOrURL, 'wb') as fp:
            fp.write(data_resp.content)

    try:
        df = pd.read_csv(fileOrURL, index_col=0)
        return df
    except:
        pass

    raise Exception('unknown data type')


def do_predict(fileOrURL, model_name):
    model_cfg = models[model_name]
    model = model_cfg['model']
    dataset_loader = model_cfg['dataset_loader']

    X = try_parse_data(fileOrURL)
    if 'similar_X_df' in model_cfg:
        similar_X_df = model_cfg['similar_X_df']
        if X.shape[1] != similar_X_df.shape[1]:
            return {'code': 1, 'msg': f'input format invalid, expect: ({X.shape[0], similar_X_df.shape[1]}), current: {X.shape}'}
        
        is_sim = is_similarity(similar_X_df, X)
        if is_sim.sum() != len(X):
            return {'code': 2, 'msg': 'Invalid spectral data.'}

    y = np.zeros(len(X)) # tmp used

    test_dataset = dataset_loader(X, y)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

    predict_prob, predict_label = misc.predict_prob_and_label(model, 'cpu', test_loader)
    if 'class_values_map' in model_cfg:
        predict_label = [model_cfg['class_values_map'].get(label, label) for label in predict_label]
    return {'code': 0, 'msg': 'success', 'predict_label': predict_label, 'predict_prob': predict_prob}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    fileOrURL = request.json.get('data_url')
    model_name = request.json.get('model_name', 'cell_exosome')
    
    try:
        result = do_predict(fileOrURL, model_name)
    except Exception as e:
        result = {'code': 3, 'msg': f'panic: {e}'}
    finally:
        print(f'request: fileOrURL={fileOrURL}, model_name={model_name}, response: {result}')
    return result

if __name__ == '__main__':
    app.run(port=5000, debug=True)
