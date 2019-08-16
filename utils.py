import  torch
import  sys


BKWD_CMPTBL_DICT = {
    'optimiser'                     :   'sgd',

    'max_train_batches_per_epoch'   :   -1,
    'max_val_batches_per_epoch'     :   -1,

    'pixel_means'                   :   [0.5, 0.5, 0.5],
    'pixel_stds'                    :   [0.5, 0.5, 0.5],
}

def fix_for_backward_compatibility(options, cmptbl_dict=BKWD_CMPTBL_DICT):
    """
    Fixes options for backward compatibility. If options are added 
    to the implementation later on, old configuration files can still
    be used by placing these new options into the variable BKWD_CMPTBL_DICT.
    They will get default values, as specified in this dictionary. 
    """
    for key in cmptbl_dict:
        val         = cmptbl_dict[key]
        if key not in options:
            if isinstance(val, dict):
                fix_backward_compatibility(options[key], cmptbl_dict=cmptbl_dict[key])
            else:
                options[key] = val

def load_state(path):
    """
    Load state dict, but do not transfer it to device, unless necessary later.
    """
    return torch.load(path, map_location=lambda s,l:s)

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
    return


def align_left(text):
    """
    Fancy stuff.
    """
    write_flush('%-70s' %(text))
    return

def write_okay():
    """
    More fancy stuff.
    """
    write_flush('[  OK  ]\n')
    return


def trim_state_dict(complete_dict, trim_key):
    """
    Trim state dict so that only those keys starting with trim_key
    remain, and the prefixed module name is removed from the key name.
    """
    trimmed     = {
                    k.replace(trim_key+'.', ''):complete_dict[k] 
                    for k in complete_dict if k.startswith(trim_key)
                  } 
    
    # The resulting dictionary can be used to load a part of a model. 
    return trimmed
