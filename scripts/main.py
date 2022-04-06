import os
import yaml
import pickle
from data_handler import DATA_HANDLER
from model import MODEL, MODEL2, MODEL3
import random
from train import train
import torch
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sacred import Experiment
from test import TEST
# from multiprocessing import Process, Pool
from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

ex = Experiment('vocal beat tracking')


@ex.config
def config():
    ex_number = 23
    train_mode = True
    device = -1
    if device < 0:
        device = 'cpu'
    else:
        device = f'cuda:{device}'

    if device == 'cpu':
        root_dir = os.path.join("C:\\", "research", "vocal_beat")
        save_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        data_dir = "C:\datasets/vocal_data/"  # cpu use
    else:
        root_dir = os.path.join("/home", "moji", "vocal_beat_server")
        save_dir = os.path.join("/storage", "moji", "vocal_beat")
        os.makedirs(root_dir, exist_ok=True)
        data_dir = "/storageNVME/moji/vocal_data"


    ex.observers.append(FileStorageObserver(save_dir + f"/results/experiment{ex_number}"))
    with open(os.path.join(root_dir, 'scripts', 'config', f'config{ex_number}.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    sample_rate = config['sample_rate']
    hop_length = int(config['hop_length'] * sample_rate)  # = 320
    seq_len = int(config['seq_len'] * sample_rate / hop_length)  # how many frames is equal to seq len in seconds
    iterations = config['iterations']
    checkpoints = config['checkpoints']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    gpu_id = config['gpu_id']
    seed = config['seed']
    num_workers_train = config['num_workers_train']
    num_workers_val = config['num_workers_val']
    num_workers_test = config['num_workers_test']
    model_type = config['model_type']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    query_dimensions = config['query_dimensions']
    value_dimensions = config['value_dimensions']
    feed_forward_dimensions = config['feed_forward_dimensions']
    estimator = config['estimator']
    data_proc = config['dataproc']
    # Add a file storage observer for the log directory


@ex.automain
def vocal_beat_run(iterations, checkpoints, batch_size, learning_rate, seed, root_dir, save_dir, data_dir,model_type,
                   seq_len, num_workers_train, num_workers_val, num_workers_test, n_layers, n_heads, query_dimensions,
                   value_dimensions, feed_forward_dimensions, estimator,ex_number, device, train_mode, data_proc):
    # Seed everything with the same seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model_type = model_type

    with open(data_dir + "/splits_ready", 'rb') as f:
        splits = pickle.load(f)

    train_data = DATA_HANDLER(splits['train'][:10], data_proc=data_proc, device=device, seq_len=seq_len, root_dir=root_dir,
                              data_dir=data_dir)
    val_data = DATA_HANDLER(splits['val'][:80], device=device, seq_len=seq_len, data_proc=data_proc, root_dir=root_dir,
                            data_dir=data_dir, random_sample=False)
    test_data = DATA_HANDLER(splits['test'], device=device, data_proc=data_proc, root_dir=root_dir, data_dir=data_dir)

    print('Loading data partitions...')
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers_train,
                              drop_last=True,
                              )

    val_loader = DataLoader(dataset=val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers_val,
                            drop_last=True)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers_test,
                             drop_last=True)
    print(data_proc)
    print('Initializing model...')
    if train_mode:
        # Initialize a new instance of the model
        if data_proc == ['wavlm']:
            model = MODEL(wavlm_type=model_type, n_layers=n_layers, n_heads=n_heads, query_dimensions=query_dimensions,
                          value_dimensions=value_dimensions, feed_forward_dimensions=feed_forward_dimensions, device=device)
        elif data_proc == ['distilhubert']:
            model = MODEL2(distil_type=model_type, n_layers=n_layers, n_heads=n_heads, query_dimensions=query_dimensions,
                          value_dimensions=value_dimensions, feed_forward_dimensions=feed_forward_dimensions, device=device)
        elif data_proc == ['log_spec']:
            model = MODEL3(n_layers=n_layers, n_heads=n_heads,query_dimensions=query_dimensions,value_dimensions=value_dimensions,
                           feed_forward_dimensions=feed_forward_dimensions,device=device)
        model.to(device)
        model.train()
        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # Decay the learning rate over the course of training
        # scheduler = StepLR(optimizer, iterations, 0.95)    # decay of learning rate

        print('Training Vocal Beat tracker...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(save_dir, 'results', f'experiment{ex_number}', 'models')

        #   Train the model
        trained = train(model=model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        iterations=iterations,
                        checkpoints=checkpoints,
                        log_dir=model_dir,
                        val_set=val_loader,
                        test_set=test_loader,  # val_loader None
                        val_estimator=estimator,
                        single_batch=False,
                        resume=False)

    else:
        print('Detection beats/downbeats and evaluating test partition...')
        trained = torch.load(r'C:\research\vocal_beat_server\results\experiment10\model/model-7.pt', map_location=torch.device('cpu'))
        import dill
        trained = dill.loads(trained)
        # plt.plot(model.sigmoid(trained.weights).detach())

        results = TEST(model=trained, dataset=test_loader, val_estimator="DBN")
    # results = test(model=trained, dataset=test_loader, val_estimator="PF",flag=1)
    # results = test(model=trained, dataset=test_loader, val_estimator="PF",flag=2)
    # get_fmeasure(beat_results, downbeat_results)
    # beat_results = np.array(beat_results)
    # downbeat_results = np.array(downbeat_results)
    # from collections import defaultdict
    # f_measures = defaultdict(list)
    # beat_f_measures = defaultdict(list)
    # down_f_measures = defaultdict(list)
    # for i in range(len(beat_results)):
    #     beat_f_measures[val_set.val_set[i].split("#")[0]].extend([beat_results[i].fmeasure])
    #     down_f_measures[val_set.val_set[i].split("#")[0]].extend([downbeat_results[i].fmeasure])
    # for i in beat_f_measures:
    #     f_measures[i] =[np.mean(beat_f_measures[i]), np.mean(down_f_measures[i])]
    # with open("C:\downbeat/results\experiment_8/f_measures", 'wb') as f:
    #     pickle.dump(f_measures, f)

    # estim_dir = os.path.join(root_dir, 'estimated')
    # results_dir = os.path.join(root_dir, 'results')

    # Get the average results for the testing partition
    # results = validate(trained, test_set, estim_dir, results_dir)

    # Log the average results in metrics.json
    # ex.log_scalar('results', results, 0)
