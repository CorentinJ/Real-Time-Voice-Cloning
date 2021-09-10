from model_vc import Generator
from generate_plot import PlotGenerator
import torch
import torch.nn.functional as F
import time
import datetime

import os
from torch.utils.tensorboard import SummaryWriter

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        
        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.resume_iters = config.resume_iters

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        
        # Directories
        self.data_dir = config.data_dir
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        self.log_step = config.log_step
        self.model_save_step = config.model_save_step


        # Build the model and tensorboard.
        self.build_model()
        self.writer = SummaryWriter(self.log_dir)

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        # self.g_lr = self.g_optimizer.param_groups[0]['lr']
    
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        data_iter = iter(data_loader)
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        
        
        # Start training.

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
    
        print('Start training...')
        start_time = time.time()
        plot_graph = True
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real_mel, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_mel, emb_org = next(data_iter)
            
            x_real_mel = x_real_mel.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()

            # plot graph once
            if(plot_graph):
                self.writer.add_graph(self.G, (x_real_mel, emb_org, emb_org))
                plot_graph = False
            
            # Identity mapping loss
            x_mel_gen, mel_gen_psnt, encoder_out = self.G(x_real_mel, emb_org, emb_org)
            # get loss on mel output vs real mel
            g_loss_id = F.mse_loss(x_real_mel, x_mel_gen)   
            g_loss_id_psnt = F.mse_loss(x_real_mel, mel_gen_psnt)   
            
            # Code semantic loss.
            # Call Generator with mel output from postnet and compare with encoder_out for loss
            code_reconst = self.G(mel_gen_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(encoder_out, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                # log for terminal
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                
                # write to tensorboard
                for tag, value in loss.items():
                    self.writer.add_scalar(tag, value, i+1)
                print("Tensorboard updated.")

                
            
            if True:#(i+1) % self.model_save_step == 0:
                files = sorted(os.listdir(self.model_save_dir), key = lambda x: int(x[:-7]))                
                file_count = len(files)
                
                # if(file_count > 2):
                #     os.remove(os.path.join(self.model_save_dir, files[0]))

                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                # torch.save({'model': self.G.state_dict(),
                            # 'optimizer': self.g_optimizer.state_dict()}, G_path)
                
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                abs_path_data = os.path.abspath(self.data_dir)
                # add visualization on male/female distrubution
                plotgen = PlotGenerator(self.G, self.writer, self.freq, i, abs_path_data, autovc=True)
                plotgen.plot_image()
                print("Added visualization on it")
                

    
    

    