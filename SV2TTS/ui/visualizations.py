from datasets.speaker_verification_dataset import SpeakerVerificationDataset
from datetime import datetime
from time import perf_counter as clock
import visdom
import params

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %H:%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.accuracy_win = None
        self.lr_win = None
        self.implementation_win = None
        self.implementation_string = ""
        self.last_step = -1
        self.last_update_timestamp = -1
        self.mean_time_per_step = -1
        self.log_params()
        
    def log_params(self):
        param_string = ""
        for param_name in (p for p in dir(params) if not p.startswith('__')):
            value = getattr(params, param_name)
            param_string += "\t%s: %s" % (param_name, value)
        param_string = param_string.replace("\n", "<br>")
        self.vis.text(param_string, opts={'title': 'Parameters'})
        
    def log_dataset(self, dataset: SpeakerVerificationDataset):
        dataset_string = ""
        for param, value in dataset.get_params().items():
            dataset_string += "<b>%s</b>: %s\n" % (param, value)
        dataset_string += "\n\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={'title': 'Dataset'})
        
    def log_implementation(self, params):
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string, 
            opts={'title': 'Training implementation'}
        )

    def update(self, loss, accuracy, lr, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )
        self.accuracy_win = self.vis.line(
            [accuracy],
            [step],
            win=self.accuracy_win,
            update='append' if self.accuracy_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Accuracy',
                title='Accuracy (mean per 10 steps)'
            )
        )
        self.lr_win = self.vis.line(
            [lr],
            [step],
            win=self.lr_win,
            update='append' if self.lr_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Learning rate',
                ytype='log',
                title='Learning rate'
            )
        )
        
        now = clock()
        if self.last_step != -1 and self.implementation_win is not None:
            time_per_step = (now - self.last_update_timestamp) / (step - self.last_step)
            if self.mean_time_per_step == -1:
                self.mean_time_per_step = time_per_step
            else:
                self.mean_time_per_step = self.mean_time_per_step * 0.9 + time_per_step * 0.1
            time_string = "<b>Mean time per step</b>: %dms" % int(1000 * self.mean_time_per_step)
            self.vis.text(
                self.implementation_string + time_string, 
                win=self.implementation_win,
                opts={'title': 'Training implementation'},
            )
        self.last_step = step
        self.last_update_timestamp = now