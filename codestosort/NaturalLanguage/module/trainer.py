import os
import sys

import torch
import logging
import pickle
import datetime
from tqdm import tqdm

sys.path.append('datapreprocess')
sys.path.append('module')
from datafunc import make_dataloader, build_processed_data, get_datas
from model import LinearNet, RnnNet, RnnAttentionNet, GruNet, LastNet
from bestmodel import BestNet
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Recall


class config():
    def __init__(self, args):
        self.datadir = '../data'
        self.pickledir = os.path.join('pick', args.pick)
        self.outputdir = os.path.join('out/', args.taskname)
        self.modelpath = os.path.join(self.outputdir, 'model.pkl')
        self.logpath = os.path.join(self.outputdir, 'log.json')

        self.modeltype_path = os.path.join(self.outputdir, 'type.json')

        self.pickle_files = ['train.pkl', 'valid.pkl', 'test.pkl', 'embedding.pkl']
        self.pickle_files = [os.path.join(self.pickledir, i) for i in self.pickle_files]

        self.start_epoch = 0
        self.epochs = args.epoch
        self.lr = 0.001
        self.grad_accumulate_steps = 3
        # now = datetime.datetime.now().strftime('%m%d%H')
        self.prediction_path = os.path.join(self.outputdir, '%s.csv'%args.modeltype)

        self.resume_epoch = args.resume_epoch


class Trainer():
    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

        logging.info('Initiating task: %s'%args.taskname)
        
        self.config = config(args)
        if not all([os.path.isfile(i) for i in self.config.pickle_files]):
            logging.info('Preprocesing data.....')
            if args.pick == 'neg4':
                build_processed_data(self.config.datadir, self.config.pickledir, neg_num=4)
            elif args.pick =='last':
                build_processed_data(self.config.datadir, self.config.pickledir, last=True)
            elif args.pick =='difemb':
                build_processed_data(self.config.datadir, self.config.pickledir, difemb=True)

            else:
                build_processed_data(self.config.datadir, self.config.pickledir)

        else: 
            logging.info('Preprocesing already done.')
        
        with open(os.path.join(self.config.pickledir, 'embedding.pkl'), 'rb') as f:
            embedding = pickle.load(f)
            embedding = embedding.vectors
        self.embedding_dim = embedding.size(1)
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        self.Modelfunc = {
            'lin': LinearNet,
            'rnn': RnnNet,
            'att': RnnAttentionNet,
            'best': BestNet,
            'gru': GruNet,
            'last': LastNet,
        }

        if os.path.exists(self.config.outputdir):
            if args.resume==False:
                logging.info('Warning, task already exists, add --resume True, exiting')
                sys.exit(0)
            else:
                logging.info('Resuming....')
                with open(self.config.modeltype_path, 'r') as f:
                    resume_type = f.read()
                self.model = self.Modelfunc[resume_type]
                logging.info('model type is %s, model to be constructed'%resume_type)
        else:
            os.mkdir(self.config.outputdir)
            with open(self.config.modeltype_path, 'w') as f:
                f.write(args.modeltype)
            self.model = self.Modelfunc[args.modeltype](self.embedding_dim)
            logging.info('model type is %s, model created'%args.modeltype)

        model_checkpoint = ModelCheckpoint(self.config.modelpath, 'loss', 1, 'all')
        metrics_logger = MetricsLogger(self.config.logpath)

        if args.resume:
            self.config.start_epoch = metrics_logger.load()
            if args.resume_epoch != -1:
                self.config.resumepath = self.config.modelpath+'.%d'%self.config.resume_epoch
            else:
                self.config.resumepath = self.config.modelpath+'.%d'%(self.config.start_epoch-1)

            self.model = self.model(self.embedding_dim)
            self.model.load_state_dict(torch.load(self.config.resumepath))
            logging.info('config loaded, model constructed and loaded')

        print(self.model)

        logging.info('loading dataloaders')
        self.trainloader, self.testloader, self.validloader = make_dataloader(self.config.pickledir)

        self.metrics = [Recall()]
        self.callbacks = [model_checkpoint, metrics_logger]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model.to(self.device)
            self.embedding.to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss()        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.lr)
    


    def run(self):
        logging.info('training for %d epochs'%self.config.epochs)

        for i in range(self.config.epochs):
            epoch = self.config.start_epoch + i
            logging.info('Training epoch {}'.format(epoch))
            log_train = self.train()
            logging.info('Validating epoch {}'.format(epoch))
            log_valid = self.valid()

            for callback in self.callbacks:
                callback.on_epoch_end(log_train, log_valid, self.model, epoch)
        self.predict()

    def run_one_batch(self, batch):
        with torch.no_grad():
            # context = self.embedding(batch['speaker'].to(self.device))
            context = self.embedding(batch['context'].to(self.device))
            # context = [self.embedding(utterance.to(self.device)) for utterance in batch['context']]
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(context, options)
        batch_loss = self.criterion(logits, batch['labels'][:,:].float().to(self.device))
        return logits, batch_loss

    def train(self):
        self.model.train()
        total = len(self.trainloader)
        trange = tqdm(enumerate(self.trainloader), total=total, desc='training')

        for metric in self.metrics:
            metric.reset()
        loss = 0
        for i, batch in trange:
            output, batch_loss = self.run_one_batch(batch)
            batch_loss /= self.config.grad_accumulate_steps
            if i % self.config.grad_accumulate_steps == 0:
                self.optimizer.zero_grad()

            batch_loss.backward()

            if (i+1)%self.config.grad_accumulate_steps ==0:
                self.optimizer.step()

            loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output, batch)

            trange.set_postfix(
                loss=loss / (i + 1),
                **{m.name: m.print_score() for m in self.metrics})

        loss /= (i+1)

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        return epoch_log

    def valid(self):
        total = len(self.validloader)
        trange = tqdm(enumerate(self.validloader),total=total, desc='validating')

        for metric in self.metrics:
            metric.reset()
        loss = 0       
        for i, batch in trange:
            output, batch_loss = self.run_one_batch(batch)

            loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output, batch)
            trange.set_postfix(
                loss=loss / (i + 1),
                **{m.name: m.print_score() for m in self.metrics})

        loss /= (i+1)

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        return epoch_log

    def predict(self):
        logging.info('Predicting to %s...'%self.config.prediction_path)
        predicts = self.predict_test()
        data = get_datas(self.config.pickledir)
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


