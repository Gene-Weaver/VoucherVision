import logging, os, psutil, torch, platform, cpuinfo, yaml #py-cpuinfo
from tqdm import tqdm
from vouchervision.general_utils import get_datetime, print_main_warn, print_main_info

class SanitizingFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        try:
            record.msg = record.msg.encode('utf-8', 'replace').decode('utf-8')
        except Exception as e:
            record.msg = f'[Error encoding text: {e}]'
        super().emit(record)

def start_logging(Dirs, cfg):
    run_name = cfg['leafmachine']['project']['run_name']
    path_log = os.path.join(Dirs.path_log, '__'.join(['LM2-log', str(get_datetime()), run_name]) + '.log')

    # Disable default StreamHandler
    logging.getLogger().handlers = [] 

    # create logger
    logger = logging.getLogger('Hardware Components')
    logger.setLevel(logging.DEBUG)

    # create custom sanitizing file handler and set level to debug
    sanitizing_fh = SanitizingFileHandler(path_log, encoding='utf-8')
    sanitizing_fh.setLevel(logging.DEBUG)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # add formatter to handlers
    sanitizing_fh.setFormatter(formatter)
    tqdm_handler.setFormatter(formatter)
    # ch.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(sanitizing_fh)
    logger.addHandler(tqdm_handler)
    # logger.addHandler(ch)

    # Create a logger for the file handler
    file_logger = logging.getLogger('file_logger')
    file_logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path_log)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    # Disable propagation of log messages to the root logger
    file_logger.propagate = False

    # 'application' code
    # logger.debug('debug message')
    # logger.info('info message')
    # logger.warning('warn message')
    # logger.error('error message')
    # logger.critical('critical message')

    # Get CPU information
    logger.info(f"CPU: {find_cpu_info()}")

    # Get GPU information (using PyTorch)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            gpu = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu.name} ({gpu.total_memory // (1024 * 1024)} MB)")
        else:
            for i in range(num_gpus):
                gpu = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {gpu.name} ({gpu.total_memory // (1024 * 1024)} MB)")
    else:
        logger.info("No GPU found")
        logger.info("LeafMachine2 image cropping and embedding search will be extremely slow or not possible.")
        print_main_info("No GPU found!")
        print_main_info("LeafMachine2 image cropping and embedding search will be extremely slow or not possible.")

    # Get memory information
    mem_info = psutil.virtual_memory()
    logger.info(f"Memory: {mem_info.total // (1024 * 1024)} MB")
    logger.info(LM2_banner())
    logger.info(f"Config added to log file")
    file_logger.info('Config:\n{}'.format(yaml.dump(cfg)))


    return logger

def find_cpu_info():
    cpu_info = []
    cpu_info.append(platform.processor())
    try:

        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_info.append(line.split(':')[1].strip())
                    break
        return ' / '.join(cpu_info)
    except:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_info = []
            cpu_info.append(info['brand_raw'])
            cpu_info.append(f"{info['hz_actual_friendly']}")
            return ' / '.join(cpu_info)
        except:
            return "CPU: UNKNOWN"

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Use tqdm's write function to ensure correct output
            self.flush()
        except Exception:
            self.handleError(record)

def LM2_banner():
        logo = """
  _                 __ __  __            _     _            ___     
 | |               / _|  \/  |          | |   (_)          |__ \    
 | |     ___  __ _| |_| \  / | __ _  ___| |__  _ _ __   ___   ) |   
 | |    / _ \/ _` |  _| |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ / /    
 | |___|  __/ (_| | | | |  | | (_| | (__| | | | | | | |  __// /_    
 |______\___|\__,_|_| |_|  |_|\__,_|\___|_| |_|_|_| |_|\___|____|   
 __      __              _    _| |_   __      ___     _             
 \ \    / /             | |  |_   _|  \ \    / (_)   (_)            
  \ \  / /__  _   _  ___| |__  |_|_ _ _\ \  / / _ ___ _  ___  _ __  
   \ \/ / _ \| | | |/ __| '_ \ / _ \ '__\ \/ / | / __| |/ _ \| '_ \ 
    \  / (_) | |_| | (__| | | |  __/ |   \  /  | \__ \ | (_) | | | |
     \/ \___/ \__,_|\___|_| |_|\___|_|    \/   |_|___/_|\___/|_| |_|"""
        return logo