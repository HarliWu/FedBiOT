import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor


class MultiLoRAAvgAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        scaler = agg_info['scaler'] if ('scaler' in agg_info) else 1.0
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def aggregate_on_model(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        scaler = agg_info['scaler'] if ('scaler' in agg_info) else 1.0
        avg_model = self._para_weighted_avg_on_model(models,
                                                     recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        if self.cfg.llm.offsite_tuning.use and \
                self.cfg.llm.offsite_tuning.save_full_model:
            ckpt = {
                'cur_round': cur_round,
                'model': self.model.state_dict(return_trainable=False)
            }
        else:
            ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None, scaler=1.0):
        """
        Calculates the weighted average of models.
        """
        keywise_training_set_size = dict()
        for i in range(len(models)):
            sample_size, model = models[i]
            for key in model.keys():
                if key not in keywise_training_set_size:
                    keywise_training_set_size[key] = [sample_size, 1]
                else:
                    keywise_training_set_size[key][0] += sample_size
                    keywise_training_set_size[key][1] += 1

        avg_model = dict()
        for i in range(len(models)):
            sample_size, model = models[i]

            for key, param in model.items():
                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / keywise_training_set_size[key][1]
                else:
                    weight = sample_size / keywise_training_set_size[key][0]

                weight = weight * scaler
                param = param2tensor(param)

                if key not in avg_model:
                    avg_model[key] = param * weight
                else:
                    avg_model[key] += param * weight

        return avg_model

    def _para_weighted_avg_on_model(self,
                                    models,
                                    recover_fun=None,
                                    scaler=1.0):
        keywise_training_size = dict()
        for i in range(len(models)):
            train_size, model = models[i]
            for key in model.keys():
                if key not in keywise_training_size:
                    keywise_training_size[key] = [train_size, 1]
                else:
                    keywise_training_size[key][0] += train_size
                    keywise_training_size[key][1] += 1

        avg_model, raw_model_scaler = dict(), dict()

        for i in range(len(models)):
            train_size, model = models[i]

            for key, param in model.items():
                if self.cfg.federate.ignore_weight:
                    if hasattr(self, 'num_clients'):
                        weight = 1.0 / self.num_clients
                    else:
                        weight = 1.0 / keywise_training_size[key][1]
                else:
                    if hasattr(self, 'total_train_size'):
                        weight = train_size / self.total_train_size
                    else:
                        weight = train_size / keywise_training_size[key][0]

                weight = weight * scaler
                param = param2tensor(param)

                if key not in avg_model:
                    avg_model[key] = param * weight
                    raw_model_scaler[key] = 1 - weight
                else:
                    avg_model[key] += param * weight
                    raw_model_scaler[key] -= weight

        # merge with the original model
        model_state_dict = self.model.state_dict()
        for key in avg_model.keys():
            avg_model[key] += raw_model_scaler[key] * model_state_dict[key]

        return avg_model
