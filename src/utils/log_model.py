from torch.utils.tensorboard import SummaryWriter
import warnings
import numpy as np

class SummaryWriterSingleton:
    _instance = None
    _log_dir = None

    def __new__(cls, log_dir=None):
        if cls._instance is not None:
            if log_dir is not None and log_dir != cls._log_dir:
                warnings.warn(
                    f"""SummaryWriterSingleton already initalised, 
                    passed log_dir will be ignored, current log_dir
                    is {cls._log_dir}""")
            return cls._instance
        
        if log_dir is not None:
            cls._instance = SummaryWriter(log_dir)
        else:
            cls._instance = SummaryWriter()
        cls._log_dir = cls._instance.log_dir

        return cls._instance
    
    def close(self):
        self.close()

def write_stats(writer, stats, step_idx):
    # print(f'writer stats {stats}')

    if ((grads := stats.get('policy_loss_grads')) is not None):
        writer.add_scalar('grad_l2', np.sqrt(np.mean(np.square(grads))), step_idx)
        writer.add_scalar('grad_max', np.max(np.abs(grads)), step_idx)
        writer.add_scalar('grad_var', np.var(grads), step_idx)
    if ((As := stats.get('As')) is not None):
        writer.add_scalar('As', As, step_idx)
    if ((Qs := stats.get('Qs')) is not None):
        writer.add_scalar('Qs', Qs, step_idx)
    if ((Vs := stats.get('Vs')) is not None):
        writer.add_scalar('Vs', Vs, step_idx)
    if ((policy_loss := stats.get('policy_losss')) is not None):
        writer.add_scalar('policy_loss', policy_loss, step_idx)
    if ((value_loss := stats.get('value_loss')) is not None):
        writer.add_scalar('value_loss', value_loss, step_idx)
    if ((entropy_bonus := stats.get('entropy_bonus')) is not None):
        writer.add_scalar('entropy_bonus', entropy_bonus, step_idx)


__all__ = ['SummaryWriterSingleton']
