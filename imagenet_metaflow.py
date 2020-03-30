from metaflow import FlowSpec, step, Parameter,IncludeFile
from constants import supported_models
import os

def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.

    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
    
class ImageNetExperimentationFlow(FlowSpec):
    '''
    A Flow to experiment and bench mark multiple ML Models in Parallel using PyTorch and Metaflow.
    '''
    zipped_dataset = IncludeFile('zipped_dataset', default=script_path('./tiny-imagenet-200.zip'),
                    help='path to ZIPPED dataset',is_text=False)
    
    zipped_dataset_name = Parameter('zipped_dataset_name',default='tiny-imagenet-200',
                            help='Name of folder Which is created when dataset is Untared')
    
    arch = Parameter('arch', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                            ' | '.join(supported_models) +
                            ' (default: resnet18)')

    workers = Parameter('workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    epochs = Parameter('epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    start_epoch = Parameter('start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    batch_size = Parameter('batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    learning_rate = Parameter('learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    momentum = Parameter('momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    weight_decay = Parameter('wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        )
    print_frequency = Parameter('pf', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    # todo : Figure Better on this 
    # Parameter('resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    
    
    evaluate = Parameter('evaluate', help='evaluate model on validation set')
    pretrained = Parameter('pretrained',
                        help='use pre-trained model')
    world_size = Parameter('world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    rank = Parameter('rank', default=-1, type=int,
                        help='node rank for distributed training')
    
    # todo : Figure Distributed Support. 
    # Parameter('dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # Parameter('dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    

    seed = Parameter('seed', default=None, type=int,
                        help='seed for initializing training. ')
    
    # Abstracting this out to use Decorators. 
    # Parameter('gpu', default=None, type=int,
    #                     help='GPU id to use.')
    multiprocessing_distributed = Parameter('multiprocessing_distributed',
                        type=bool,
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    @step
    def start(self):
        # todo : define Hyper Param Search Class
        self.trained_on_gpu = False
        self.used_num_gpus = 0 
        self.next(self.train_model)


    @step
    def train_model(self):
        from zipfile import ZipFile
        from io import BytesIO
        import random 
        import imagenet_pytorch
        # Create Directory for deloading dataset. 
        random_hex = str(hex(random.randint(0,16777215)))
        self.dataset_final_path = script_path('dataset-'+random_hex)
        safe_mkdir(self.dataset_final_path)

        # Create the Directory for the dataset using Zip Provided as the input dataset.
        zipdata = BytesIO()
        zipdata.write(self.zipped_dataset)
        dataset_zip_file = ZipFile(zipdata)
        
        # Extract the dataset 
        dataset_zip_file.extractall(self.dataset_final_path)
        print("Extracted Dataset")
        self.dataset_final_path = os.path.join(self.dataset_final_path,self.zipped_dataset_name)
        
        imagenet_pytorch.main(self)
        self.next(self.end)

        
    @step
    def end(self):
        pass
        # self.next(self.last)

if __name__ == '__main__':
    ImageNetExperimentationFlow()
