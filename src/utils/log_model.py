from torch.utils.tensorboard import SummaryWriter
import warnings

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


__all__ = ['SummaryWriterSingleton']
