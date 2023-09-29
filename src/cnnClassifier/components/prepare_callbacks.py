import os
import time
import torch 
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from cnnClassifier.entity.config_entity import PrepareCallbackConfig


class PrepareCallback:
    def __init__(self, config : PrepareCallbackConfig):
        """
        This function initializes the class
    
        Parameters
        ----------
        config : PrepareCallbackConfig
            config file
        
        Returns
        -------
        None
        """
        self.config = config
    
    @property
    def _create_tb_callbacks(self):
        """
        This function creates the tensorboard callbacks
        
        Parameters
        ----------
        None    
    
        Returns
        ------- 
        SummaryWriter
            tensorboard callback
        """
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        return SummaryWriter(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self, model : nn.Module, optimizer : optim , epoch : int, loss : float):
        """
        This function creates the checkpoint callbacks  
    
        Parameters
        ----------  
        model : nn.Module
            model
        optimizer : torch.optim 
            optimizer
        epoch : int 
            epoch number    
        loss : float    
            loss value
        
        Returns 
        -------
        None
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.config.checkpoint_model_filepath)

        
    def get_tb_ckpt_callbacks(self, model : nn.Module = None , optimizer : optim = None , epoch : int = None, loss : float = None) -> list:
        """
        This function creates the tensorboard and checkpoint callbacks  
        
        Parameters
        ----------  
        model : nn.Module
            model
        optimizer : torch.optim 
            optimizer
        epoch : int 
            epoch number
        loss : float    
            loss value
        
        Returns
        -------
        list
            list of callbacks
        """
        return [
            self._create_tb_callbacks, 
            lambda: self._create_ckpt_callbacks(model, optimizer, epoch, loss)
        ]
        
