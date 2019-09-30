import os
import sys

import torch
import logging
import pickle
import datetime
from tqdm import tqdm

sys.path.append('datapreprocess')
sys.path.append('module')
from datafunc import make_dataloader, test_data_for_predict, build_processed_data, make_validloader
from model import LinearNet, RnnNet, RnnAttentionNet

class config():
    def __init__(self):
        self.datadir = '../data'
        self.outputdir = 'out/nltk'
        self.pickle_files = ['train.pkl', 'valid.pkl', 'test.pkl', 'embedding.pkl']
        self.pickle_files = [os.path.join(self.outputdir, i) for i in self.pickle_files]

        self.epochs = 10
        self.lr = 0.001
        now = datetime.datetime.now().strftime('%m%d%H')
        self.prediction_path = '%s.csv'%now

class Trainer():
    def __init__(self):
        self.config = config()
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')



        if not all([os.path.isfile(i) for i in self.config.pickle_files]):
            logging.info('Preprocesing data.....')
            build_processed_data(self.config)
        else: 
            logging.info('Preprocesing already done.')

        # get dataloader
        # self.trainloader, self.testloader, self.validloader = make_dataloader(self.config.outputdir)
        valid_data, validloader =  make_validloader(self.config.outputdir)
        # build model
        self.load_embedding()

        for batch in validloader:
            # print(batch)
            print(batch['context'].shape)
            print(batch['options'].shape)
            # print(batch['context'])
            print(self.embedding(batch['context']).shape)
            print(self.embedding(batch['options']).shape)
            break


        # self.model = LinearNet(self.embedding_dim)

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if self.device == 'cuda':
        #     self.model.to(self.device)
        #     self.embedding.to(self.device)

        # self.criterion = torch.nn.BCEWithLogitsLoss()        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), 
        #                                   lr=self.config.lr)
    
    def load_embedding(self):
        with open(os.path.join(self.config.outputdir, 'embedding.pkl'), 'rb') as f:
            embedding = pickle.load(f)
            embedding = embedding.vectors
        self.embedding_dim = embedding.size(1)
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

    def run(self):
        for epoch in range(self.config.epochs):
            logging.info('Training epoch {}'.format(epoch))
            self.train()
            logging.info('Validating epoch {}'.format(epoch))
            self.valid()
        self.predict()

    def train(self):
        for batch in tqdm(self.trainloader, desc='training'):
            loss, output = self.run_one_batch(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def valid(self):
        for batch in tqdm(self.trainloader, desc='validating'):
            loss, output = self.run_one_batch(batch)

    def run_one_batch(self, batch):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        output = self.model.forward(context, options)
        loss = self.criterion(output, batch['labels'][:,:].float().to(self.device))
        return loss, output


    def predict(self):
        logging.info('Predicting...')
        predicts = self.predict_test()
        print(len(predicts))
        data = test_data_for_predict(self.config.outputdir)
        self.write_predict_csv(predicts, data, self.config.prediction_path)


    def predict_test(self):
        self.model.eval()
        ys_ = []
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc = 'predicting'):
                batch_y_ = self._predict_batch(batch)
                ys_.append(batch_y_)
        ys_ = torch.cat(ys_, 0)
        return ys_

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            # batch['context_lens'],
            options.to(self.device),
            # batch['option_lens']
            )
        return logits

    @staticmethod
    def write_predict_csv(predicts, data, output_path, n=10):
        outputs = []
        for predict, sample in zip(predicts, data):
            candidate_ranking = [
                {
                    'candidate-id': oid,
                    'confidence': score.item()
                }
                for score, oid in zip(predict, sample['option_ids'])
            ]

            candidate_ranking = sorted(candidate_ranking,
                                       key=lambda x: -x['confidence'])
            best_ids = [candidate_ranking[i]['candidate-id']
                        for i in range(n)]
            outputs.append(
                ''.join(
                    ['1-' if oid in best_ids
                     else '0-'
                     for oid in sample['option_ids']])
            )

        logging.info('Writing output to {}'.format(output_path))
        with open(output_path, 'w') as f:
            f.write('Id,Predict\n')
            for output, sample in zip(outputs, data):
                f.write(
                    '{},{}\n'.format(
                        sample['id'],
                        output
                    )
                )


