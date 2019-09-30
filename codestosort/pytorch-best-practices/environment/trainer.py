## import model
## import configs

class Trainer():
	def __init__(self, config, data_loader):
		self.config = config

		self.data_loader = data_loader

		## build model
		self.model = Model()
		print('[*] Number of model parameters: {:,}'.format(
	        sum([p.data.nelement() for p in self.model.parameters()])))

	def train(self):

		#resume

		for epoch in epochs:
			print()
			train_items = self.train_one_epoch(epoch)
			valid_items = self.validate(epoch)

		self.save_checkpoint()


	def train_one_epoch(self):

	def validate(self):

	def test(self):

	def save_checkpoint(self):

	def load_checkpoint(self):
		