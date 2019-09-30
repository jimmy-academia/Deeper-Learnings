import math
import json


class Callback:
    def __init__():
        pass

    def on_epoch_end(log_train, log_valid, model):
        pass


class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {
            'train': [],
            'valid': []
        }
        self.log_dest = log_dest

    def load(self):
        with open(self.log_dest,'r') as f:
            self.history = json.load(f)
        return len(self.history['train'])

    def on_epoch_end(self, log_train, log_valid, model, epoch):
        log_train['epoch'] = epoch
        log_valid['epoch'] = epoch
        # log_train['Recall@10'] = log_train['Recall@10'].item()
        # log_valid['Recall@10'] = log_valid['Recall@10'].item()
        self.history['train'].append(log_train)
        self.history['valid'].append(log_valid)
        with open(self.log_dest, 'w') as f:
            json.dump(self.history, f, indent='    ')


class ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='loss',
                 verbose=0,
                 mode='min'):
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode

    def on_epoch_end(self, log_train, log_valid, model, epoch):
        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

        elif self._mode == 'max':
            if score > self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

        elif self._mode == 'all':
            model.save('{}.{}'
                       .format(self._filepath, epoch))
