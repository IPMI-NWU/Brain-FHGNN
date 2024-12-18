class ModifiedArgs(object):
    def __init__(self, name_space, update_dict):
        name_space_dict = vars(name_space)
        for key in name_space_dict:
            setattr(self, key, name_space_dict[key])

        for key in update_dict:
            value = update_dict[key]
            try:
                value = int(value)
            # due to this bug learning rate will be always set to zero as it is converted to int for nni tunning
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            setattr(self, key, value)
