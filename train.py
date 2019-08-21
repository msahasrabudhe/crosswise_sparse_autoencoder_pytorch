"""
Training script for Pytorch implementation of Crosswise Sparse Convolutional Autoencoder. 
See https://www3.cs.stonybrook.edu/~cvl/content/papers/2018/Le_PR19.pdf
"""

from    models      import  *
from    datasets    import  *

import  argparse

import  time

def main(sys_string=None):
    """
    Main function for the train script. 
    Can be called from the python interpreter by passing a custom string
    argument. For example

    >>> main('--cfg configs/foo.yaml --gpu 0')

    When called from cmd, uses the command-line arguments instead.
    """
    parser              = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str, help='Path to configuration file.')
    parser.add_argument('--gpu', default=-1, type=int, help='Which GPU to use. A value of -1 signifies no CUDA.')
    parser.add_argument('--output_dir', default='output/', type=str, help='Path to directory which will store output.')

    if sys_string is not None:
        args            = parser.parse(sys_string.split(' '))
    else:
        args            = parser.parse_args()

    assert os.path.exists(args.cfg), 'Specified cfg file {} does not exist!'.format(args.cfg)

    # Experiment parameters are found in the .yaml configuration file. 
    options             = load_yaml(args.cfg)
    fix_for_backward_compatibility(options)

    assert options.optimiser in ['sgd', 'adam'], 'Only Adam and SGD optimisers handled for now. Got {}'.format(options.optimiser)

    assert args.gpu == -1 or args.gpu < torch.cuda.device_count(), \
            '--gpu must be either -1 or must specify a valid GPU ID (num_gpus={}). Got {}'.format(torch.cuda.device_count(), args.gpu)

    if args.gpu == -1:
        device          = 'cpu'
    else:
        device          = 'cuda:%d' %(args.gpu)

    # Class to use to define the dataset. This is defined in datasets.py
    DatasetClass        = PatchCamelyonImageset

    # Get the experiment name. This will be used to determine the output directory. 
    __, cfgname         = os.path.split(args.cfg)
    exp_name            = cfgname.replace('.yaml', '')

    # Create sub-directories inside the output directory to record intermediate results. 
    imgs_to_record      = ['input', 'recon', 'foreground', 'background', 'detection_map']

    # Get output directory. 
    output_dir          = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_dir):
        # If the output directory does not exist, we create it,
        #   as well as all the sub-directories. 
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images/'))
        for dname in imgs_to_record:
            os.makedirs(os.path.join(output_dir, 'images/', dname))
    else:
        # If the output directory already exists, check whether a previous training session is recorded. 
        # In case it is recorded, we will resume that session instead of starting from a fresh model.
        if os.path.exists(os.path.join(output_dir, 'system_state.pkl')):
            # Training is only saved every epoch. Presence of system_state.pkl signifies at least
            #   one epoch of training has passed.
            print('Found previous training instance in %s. Will attempt to resume using the previously saved state.' %(output_dir))
            # By setting this variable, we force loading of the previously-saved model. 
            options.load    = output_dir

        # Check for sub-directories, just in case. 
        try:    os.makedirs(os.path.join(output_dir, 'images/'))
        except: pass

        for dname in imgs_to_record:
            try:    os.makedirs(os.path.join(output_dir, 'images/', dname))
            except: pass

    align_left('Initialising model ...')
    model               = CrosswiseSparseCAE(options).to(device)
    if options.optimiser == 'sgd':
        optimiser       = optim.SGD(model.parameters(), lr=options.lr, weight_decay=options.weight_decay, momentum=options.momentum)
    elif options.optimiser == 'adam':
        optimiser       = optim.Adam(model.parameters(), lr=options.lr, weight_decay=options.weight_decay, betas=(options.momentum, 0.99))
    write_okay()
    print(model)

    epoch               = 0
    iter_mark           = 0
    train_history       = []
    val_history         = []
    iter_history        = []

    # If we have to load a previously saved model ...
    if options.load != '':
        align_left('Loading previously saved model ...')
        assert os.path.exists(os.path.join(options.load, 'net.pth')),       'No saved model found at {}!'.format(os.path.join(options.load, '_net.pth'))
        assert os.path.exists(os.path.join(options.load, 'optimiser.pth')), 'No saved optimiser found at {}!'.format(os.path.join(options.load, 'optimiser.pth'))

        model.load_state_dict(load_state(os.path.join(options.load, 'net.pth')))
        optimiser.load_state_dict(load_state(os.path.join(options.load, 'optimiser.pth')))

        with open(os.path.join(options.load, 'system_state.pkl'), 'rb') as fp:
            system_state    = pickle.load(fp)
        train_history   = system_state['train_history']
        val_history     = system_state['val_history']
        iter_history    = system_state['iter_history']
        epoch           = system_state['epoch']
        iter_mark       = system_state['iter_mark']
        write_okay()

    # Fix patch size. To be used for random crop. 
    if options.patch_size == -1:
        patch_size      = options.image_size
    else:
        patch_size      = options.patch_size

    # Transforms for the training dataset. 
    train_transforms    = vtransforms.Compose([
                            vtransforms.RandomCrop(patch_size, pad_if_needed=True, padding_mode='reflect'),
                            RandomRotation([0, 90, 180, 270]),
                            vtransforms.RandomHorizontalFlip(p=0.5),
                            vtransforms.RandomVerticalFlip(p=0.5),
                          ])


    align_left('Building datasets ...')
    # The val dataset does not have any extra transforms associated with it. 
    train_dataset       = DatasetClass(options, splits=['train'], img_transforms=train_transforms)
    val_dataset         = DatasetClass(options, splits=['val'], img_transforms=None)
    write_okay()

    # Loss scaling. An option scale_<loss_name> in the .yaml specifies
    #   the scaling hyperparameter for this loss. If such an option is not
    #   present, a default value of 1 is used. 
    loss_scaling_dict   = {}
    for _loss in options.losses:
        if 'scale_'+_loss in options:
            loss_scaling_dict[_loss] = options['scale_'+_loss]
        else:
            loss_scaling_dict[_loss] = 1.0


    # Closure to compute forward pass of the model on a data point. 
    # Specifying closures results in clearer code with no repetitions. 
    # Editing code is easy because edits have to be made in only one
    #   place. 
    def closure(data_point, train=True):
        # Compute the loss on a data point. 
        img             = data_point[0].to(device)

        # Compute forward pass. 
        return_dict     = model(img, train=train)

        # recon stores the reconstruction. 
        recon           = return_dict['recon']

        # Define the reconstruction loss. 
        loss_definitions = {
            'recon'         : lambda : F.mse_loss(recon, img), 
        }

        # Compute all losses that are mentioned in options.losses. 
        # This makes adding losses very easy, as to add a new loss, 
        #   only its definition needs to be added to the dictionary 
        #   above. 
        losses          = [ loss_scaling_dict[_loss] * loss_definitions[_loss]() \
                            for _loss in options.losses ]

        # Return the input as well (useful in validation phase to save images).
        return_dict['input']    = img

        return losses, return_dict

    # Execute one epoch on the specified dataloader. 
    def dataloader_epoch(loader, ep, train=True):
        # Global variables. Need to be updated, hence nonlocal.
        nonlocal iter_mark
        nonlocal iter_history

        # This list stores values of all losses that appear in options.losses
        loss_indiv      = [0 for _l in options.losses]
        n_imgs          = 0

        # Start time to record elapsed time and compute ETA.
        start_time      = time.time()

        # Maximum batches to execute in this epoch. 
        max_batches     = options.max_train_batches_per_epoch if train else options.max_val_batches_per_epoch
        final_iter      = len(loader) if max_batches == -1 else max_batches

        # Iterate on the dataloader. 
        for batch_idx, data_point in enumerate(loader, 0):
            batch_size  = data_point[0].size(0)
            n_imgs     += batch_size

            # Reset gradients. 
            optimiser.zero_grad()

            # Execute the closure. 
            losses, ret = closure(data_point, train=train) 

            # Total loss. 
            loss        = sum(losses)

            # Images to record. 
            d_map       = ret['detection_map']
            tau         = model.detection_branch.tau.item()
            sparsity    = d_map.mean().item()

            # Data point to store iteration history. 
            hist_data_point = {}
            hist_data_point['total'] = loss.item()
            hist_data_point['tau'] = tau
            hist_data_point['sparsity'] = sparsity

            # Time taken. Helpful. 
            cur_time    = time.time()
            elapsed     = cur_time - start_time
            avg_per_it  = elapsed / (batch_idx + 1)
            remain      = final_iter - batch_idx - 1
            ETA         = remain * avg_per_it

            elapsed_m   = np.int(elapsed) // 60
            elapsed_s   = elapsed - 60 * elapsed_m
            ETA_m       = np.int(ETA) // 60
            ETA_s       = ETA - 60 * ETA_m

            write_flush('Elapsed: %02dm%02ds .. ETA: %02dm%02ds | '%(elapsed_m, elapsed_s, ETA_m, ETA_s))
            if train:
                write_flush('Epoch %d, batch %d of %d, iteration %d' %(ep, 1+batch_idx, final_iter, iter_mark))
            else:
                write_flush('Batch %d of %d' %(batch_idx+1, len(loader)))

            write_flush('  .. total: %.4f' %(loss.item()))
            for _l in range(len(options.losses)):
                _loss   = options.losses[_l]
                _val    = losses[_l].item()

                hist_data_point[_loss] = _val
                write_flush(' .. %s: %.4f' %(_loss, _val))
                loss_indiv[_l] += batch_size * _val

            write_flush(' | tau = %g, sparsity = %g' %(tau, sparsity))
            write_flush('\n')

            if train:
                loss.backward()
                optimiser.step()
                iter_mark  += 1
                iter_history.append(hist_data_point)

            # Train only these many batches every epoch. 
            if batch_idx + 1 == final_iter:
                break
            
        # Divide by the number of images used to get average loss per image. 
        for _l in range(len(loss_indiv)):
            loss_indiv[_l] /= n_imgs

        elapsed         = time.time() - start_time
        elapsed_m       = np.int(elapsed) // 60
        elapsed_s       = elapsed - 60 * elapsed_m
        avg_m           = np.int(elapsed / final_iter) // 60
        avg_s           = elapsed / final_iter - 60 * avg_m
        write_flush('Time taken for this epoch: %02dm%02ds. Average time per iteration: %02dm%7.4fs\n' %(elapsed_m, elapsed_s, avg_m, avg_s))

        return loss_indiv, ret


    # Loop to run for n_epochs. 
    while epoch < options.n_epochs:
        # Decay learning rate. We decay it here so that it is easier to resume from a saved training session.
        if epoch in options.lr_decay_at:
            align_left('Decaying learning rates ...')
            for opt in [optimiser]:
                for pg in opt.param_groups:
                    pg['lr']       *= options.lr_decay
            write_okay()


        #   ==============
        #   Train phase.
        model.train()
        print('Training ...')

        epoch_start         = time.time()

        train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.workers)
        train_losses, __    = dataloader_epoch(train_loader, epoch, train=True)
        l_train_total       = sum(train_losses)

        train_hist_data_point = {}
        train_hist_data_point['total'] = l_train_total
        for _loss, _val in zip(options.losses, train_losses):
            train_hist_data_point[_loss] = _val
        train_history.append(train_hist_data_point)
        #   ==============


        #   ==============
        #   Val phase
        model.eval()
        print('\nValidating ...')
    
        val_loader          = torch.utils.data.DataLoader(val_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.workers)
        with torch.no_grad():
            val_losses, ret = dataloader_epoch(val_loader, epoch, train=False)
        l_val_total         = sum(val_losses)

        val_hist_data_point = {}
        val_hist_data_point['total'] = l_val_total
        for _loss, _val in zip(options.losses, val_losses):
            val_hist_data_point[_loss] = _val
        val_history.append(val_hist_data_point)
        #   ==============

        epoch_end           = time.time()


        system_state    = {
            'train_history'         : train_history,
            'val_history'           : val_history, 
            'iter_history'          : iter_history,
            'epoch'                 : epoch + 1,
            'iter_mark'             : iter_mark,
        }
        
        #   ==============
        print('\nEpoch %d summary' %(epoch))
        print('===================================')
        print('Train: ')
        for _loss in ['total'] + options.losses:
            write_flush('\t%15s: %.4f' %(_loss, train_hist_data_point[_loss]))
        print('\nVal: ')
        for _loss in ['total'] + options.losses:
            write_flush('\t%15s: %.4f' %(_loss, val_hist_data_point[_loss]))
        print('\nTotal time taken (train + val): %.4fs' %(epoch_end - epoch_start))
        print('\n===================================')

        with open(os.path.join(output_dir, 'system_state.pkl'), 'wb') as fp:
            pickle.dump(system_state, fp)
       
        align_left('Saving models ...')
        torch.save(model.state_dict(), os.path.join(output_dir, 'net.pth'))
        torch.save(optimiser.state_dict(), os.path.join(output_dir, 'optimiser.pth'))
        if (epoch + 1) in options.checkpoint_special:
            try:
                os.makedirs(os.path.join(output_dir, '%d'%(epoch)))
            except:
                pass

            torch.save(model.state_dict(), os.path.join(output_dir, '%d'%(epoch), 'net.pth'))
            torch.save(optimiser.state_dict(), os.path.join(output_dir, '%d'%(epoch), 'optimiser.pth'))
        write_okay()

        print('--> Epoch %d model, optimiser, and system state saved to %s.' %(epoch, output_dir))

        # Save images. 
        # First, find a suitable number of rows. 
        _bs                 = ret['input'].size(0)
        _sq                 = int(np.floor(np.sqrt(_bs)))
        while _bs % _sq != 0:
            _sq            += 1

        align_left('Saving images from result ...')
        tensors_to_record   = [ret[_name].detach().cpu() for _name in imgs_to_record]
        for T, _name in zip(tensors_to_record, imgs_to_record):
            filename        = os.path.join(output_dir, 'images/', _name, '%06d.png' %(epoch))
            vutils.save_image(T, filename, nrow=_sq, padding=2, normalize=True)
        write_okay()

        epoch          += 1

if __name__ == '__main__':
    main()
