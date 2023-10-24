import os
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel

# from botorch.models.utils import fantasize as fantasize_flag
from botorch import settings
from deepopt.surrogate_utils import create_optimizer
from deepopt.surrogate_utils import MLP as Arch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeltaEnc(BaseModel):
    def __init__(self,
                 network,
                 config,
                 optimizer,
                 X_train,
                 y_train,
                 surrogate_type,
                 encoding='default',
                 target='dy',  # 'y', 'dy'
                 device='cpu',
                 scale=1
                 ):

        super().__init__()
        if isinstance(network,list):
            self.multi_network = True
            self.n_epochs = [cf['n_epochs'] for cf in config]
            self.actual_batch_size = [min(cf['batch_size'], len(X_train)) for cf in config]
        else:
            self.multi_network = False
            self.n_epochs = config['n_epochs']
            self.actual_batch_size = min(config['batch_size'], len(X_train))
        
        self.f_predictor = network
        self.f_optimizer = optimizer
        self.config = config
        
        X_train = X_train.float()
        y_train = y_train.float()

        self.X_train = X_train
        self.y_train = y_train
        
        self.input_dim = X_train.shape[-1]
        self.output_dim = y_train.shape[-1]
                
        self.is_fantasy_model = False
        
        batch_shape = X_train.shape[:-2]
        self._batch_shape = batch_shape
        self.n_train = X_train.shape[-2]
        
        # self.train_fid_locs = [X_train[...,-1]==i for i in X_train[...,-1].unique()]
        # y_train_by_fid = [y_train[fid_loc] for fid_loc in self.train_fid_locs]
        
        # self.y_max = [Y.max(dim=0)[0] for Y in y_train_by_fid]
        # self.y_min = [Y.min(dim=0)[0] for Y in y_train_by_fid]
        # self.y_max = y_train.max(dim=0)[0]
        # self.y_min = y_train.min(dim=0)[0]
        self.y_max = y_train.max().detach()
        self.y_min = y_train.min().detach()
        self.out_scaler = lambda y,y_min,y_max:(y-y_min)/(y_max-y_min)*scale
        self.out_scaler_inv = lambda y,y_min,y_max: y*(y_max-y_min)/scale+y_min
        
        self.X_train_scaled = X_train.clone()
        self.y_train_scaled = self.out_scaler(y_train.clone(),self.y_min,self.y_max)
        
        
        # for fid_loc,y_min,y_max in zip(self.train_fid_locs,self.y_min,self.y_max):
        #     self.y_train_scaled[fid_loc] = self.out_scaler(self.y_train_scaled[fid_loc],y_min,y_max)
                
        self.X_train_nn = self.X_train_scaled.moveaxis(-2,0).reshape(self.n_train,-1)
        self.y_train_nn = self.y_train_scaled.moveaxis(-2,0).reshape(self.n_train,-1)
        
        self.nn_input_dim = self.X_train_nn.shape[1]
        self.nn_output_dim = self.y_train_nn.shape[1]
        
        # self.anchor_X = [X_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        # self.anchor_Y = [y_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        # self.y_max = [Y.max(axis=0)[0] for Y in self.anchor_Y]
        # self.y_min = [Y.min(axis=0)[0] for Y in self.anchor_Y]
        # self.out_scaler = lambda y,y_min,y_max:(y-y_min)/(y_max-y_min)*scale
        # self.out_scaler_inv = lambda y,y_min,y_max: y*(y_max-y_min)/scale+y_min
        
        # self.X_train = torch.cat(self.anchor_X,axis=0)
        # self.anchor_Y = [self.out_scaler(y,y_min,y_max) for (y,y_min,y_max) in zip(self.anchor_Y,self.y_min,self.y_max)]
        # self.y_train = torch.cat(self.anchor_Y,axis=0)

        self.loss_fn = nn.MSELoss()

        self.device = device
        self.surrogate_type = surrogate_type
        self.target = target
        self.encoding = encoding

        # define this member variable or botorch will not 
        # allow transformations to be applied to inputs
        #self.batch_shape = batch_shape
        # if batch_shape>1:
        #     self.train_inputs = self.X_train.expand(batch_shape,*self.X_train.shape)
        # else:
        self.train_inputs = self.X_train
        if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
            self.train_inputs = (self.train_inputs,)
            # assert self.train_inputs.shape[0]%batch_shape==0, 'Number of training samples ({}) must be divisible by batch_shape ({})'.format(self.train_inputs.shape[0],batch_shape)
            # samples_in_batch = self.train_inputs.shape[0]//batch_shape
            # self.train_inputs = tuple(self.train_inputs[i*samples_in_batch:(i+1)*samples_in_batch] for i in range(batch_shape))
            
        
    @property
    def batch_shape(self):
        return self._batch_shape
    
    @batch_shape.setter
    def batch_shape(self,value):
        self._batch_shape = value
        
    # coped from model.py
    # Originally, this method accessed the first dim of self.train_inputs (i.e self.train_inputs[0])
    # because the inputs are assumed to be in batches.
    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    # copied from exact_gp.py
    # This needs to be defined if you are passing in an input transformation.
    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
            if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
                self.train_inputs = (self.train_inputs,)            
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def fit(self):
        
        if self.multi_network:
            pass
        else:
            data = TensorDataset(self.X_train_nn, self.y_train_nn)
            loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
            self.f_predictor.train()
            # pbar = tqdm(range(self.n_epochs), desc='train epochs', leave=False)
            for epoch in range(self.n_epochs):
                avg_loss = 0.0

                for batch_id, (xi, yi) in enumerate(loader):
                    xi = xi.to(self.device)
                    yi = yi.to(self.device)
                    
                    # xi_fid = [xi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]
                    # yi_fid = [yi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]

                    if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
                        xi = self.f_predictor.input_mapping(xi)
                    #     xi_fid = [self.f_predictor.input_mapping(xi) for xi in xi_fid]
                    #     X_anchors = [self.f_predictor.input_mapping(anchor) for anchor in self.anchor_X]
                    # else:
                    #     X_anchors = [anchor for anchor in self.anchor_X]
                            
                    flipped_x = torch.flip(xi, [0])
                    if self.encoding == 'default':
                        diff_x = xi-flipped_x
                        inp = torch.cat([flipped_x, diff_x], axis=1)

                    if self.target == 'y':
                        out = yi
                    else:
                        flipped_y = torch.flip(yi, [0])
                        if self.encoding == 'default':
                            diff_y = yi-flipped_y
                            out = diff_y

                    out_hat = self.f_predictor(inp)
                    self.f_optimizer.zero_grad()
                    f_loss = self.loss_fn(out_hat.float(), out.float())
                    f_loss.backward()
                    self.f_optimizer.step()
                    avg_loss += f_loss.item()/len(loader)
                # if epoch % 25 == 0:
                #     pbar.set_description('Error %f' % avg_loss)
                # print('Epoch {} Avg.train loss {}'.format(epoch, avg_loss))
                

    def save_ckpt(self, path, name):
        state = {'epoch': self.n_epochs}
        state['state_dict'] = self.f_predictor.state_dict()
        state['B'] = self.f_predictor.B
        state['opt_state_dict'] = self.f_optimizer.state_dict()
        filename = path + '/' + name + '.ckpt'
        torch.save(state, filename)
        print('Saved Ckpts')

    def load_ckpt(self, path, name):
        saved_state = torch.load(os.path.join(
            path, name + '.ckpt'), map_location=self.device)
        self.f_predictor.load_state_dict(saved_state['state_dict'])
        self.f_predictor.B = saved_state['B']

    def _map_delta_model(self, ref, query):
        if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
            ref = self.f_predictor.input_mapping(ref)
            query = self.f_predictor.input_mapping(query)
        if self.encoding == 'default':
            diff = query - ref
        samps = torch.cat([ref, diff], 1)
        pred = self.f_predictor(samps)

        return pred

    def get_prediction_with_uncertainty(self, q,get_cov=False,original_scale=True):
        orig_input_shape = q.shape
        assert q.shape[-1]==self.input_dim, 'Expected tensor to have size=input_dim ({}) in last dimension, found tensor of shape {}'.format(self.input_dim,q.shape)        
        if q.shape[-len(self.batch_shape)-2:-2]!=self.batch_shape:
            try:
                print('Need to expand input shape {shape} to match training batch shape {batch_shape}.'.format(shape=orig_input_shape,batch_shape=self.batch_shape))
                if q.shape[-len(self.batch_shape)-2:-2]==torch.Size(len(self.batch_shape)*[1]):
                    q = q.expand(*q.shape[:-len(self.batch_shape)-2],*self.batch_shape,*q.shape[-2:])
                else:
                    q = q.expand(*self.batch_shape,*q.shape)
                    for _ in range(len(self.batch_shape)):
                        q = q.moveaxis(0,-3)
            except Exception as e:
                print('Could not make tensor of shape {shape} match training batch shape {batch_shape}.'.format(shape=orig_input_shape,batch_shape=self.batch_shape))
                print(e)
        assert q.shape[-len(self.batch_shape)-2:-2]==self.batch_shape, 'Expected tensor to have batch shape matching training batch shape {batch_shape}, instead found tensor of shape {shape}.'.format(batch_shape=self.batch_shape,shape=q.shape)
        input_shape = q.shape
        #n_test = q.shape[-2]
        # test_fid_locs = [q[...,-1]==i for i in self.X_train[...,-1].unique()]
        
        q_move = q.moveaxis(-2,0)
        samples_shape = q_move.shape[:-len(self.batch_shape)-1]
        
        n_ref = 100
        # ref_choice = np.random.choice(self.X_train_nn.shape[0],n_ref*int(np.prod(samples_shape[1:])))
        ref_choice = torch.randint(self.X_train_nn.shape[0],(n_ref*int(np.prod(samples_shape[1:])),))
        ref = self.X_train_nn[ref_choice].reshape(n_ref,1,*samples_shape[1:],self.X_train_nn.shape[-1]).expand(n_ref,*samples_shape,self.X_train_nn.shape[-1]).reshape(-1,self.X_train_nn.shape[-1])
        ref_y = self.y_train_nn[ref_choice].reshape(n_ref,1,*samples_shape[1:],self.y_train_nn.shape[-1]).expand(n_ref,*samples_shape,self.y_train_nn.shape[-1]).reshape(-1,self.y_train_nn.shape[-1])

        q_combine_samples = q_move.expand(n_ref,*q_move.shape).reshape(-1,*self.batch_shape,self.input_dim)
        q_reshape = q_combine_samples.reshape(q_combine_samples.shape[0],-1)
        # n_test = q_reshape.shape[0]
        # n_ref = 100
        # all_preds = torch.zeros(nref,*q.shape[:-1],1) if q.shape[-1]==self.input_dim else torch.zeros(nref,*q.shape[:-2],1,1)
        #all_preds = torch.zeros(n_ref,self.nn_input_dim,self.nn_output_dim)
        self.f_predictor.eval()
        # ref = self.X_train_nn[np.random.choice(self.X_train_nn.shape[0],n_ref*n_test)].to(self.device)
        # ref_y = self.y_train_nn[np.random.choice(self.y_train_nn.shape[0],n_ref*n_test)].to(self.device)
        # val = self._map_delta_model(ref,q_reshape.float().tile(dims=(n_ref,1)))
        val = self._map_delta_model(ref,q_reshape.float())
        if self.target != 'y':
            val+=ref_y
            
        val = val.reshape(n_ref,*samples_shape,*self.batch_shape,self.output_dim).moveaxis(1,-2)
        assert val.shape[1:-1]==input_shape[:-1], 'Something went wrong with reshaping.'
        # if collapse_flag:
        #     for _ in range(len(self.batch_shape)):
        #         val = val.moveaxis(-3,0).mean(axis=0)
        # assert val.shape[1:-1]==orig_input_shape[:-1], 'Something went wrong with collapsing.'
            
        if original_scale:
            all_preds = self.out_scaler_inv(val,self.y_min,self.y_max)
            # all_preds = torch.zeros_like(val)
            # for fid_loc,ymin,ymax in zip(test_fid_locs,self.y_min,self.y_max):
            #     all_preds[:,fid_loc] = self.out_scaler_inv(val[:,fid_loc],ymin,ymax)
        else:
            all_preds = val
                        
        mu = all_preds.mean(axis=0)
        var = all_preds.var(axis=0)
        if get_cov:
            assert self.output_dim==1, 'Output must be 1-dimensional to compute covariances.'
            all_preds = all_preds.squeeze(-1)
            mu = mu.squeeze(-1)
            var = var.squeeze(-1)
            del_pred = all_preds - all_preds.mean(axis=0)
            cov = torch.einsum('i...A,i...B->...AB',del_pred,del_pred)/(n_ref-1)
            cov = 0.5*(cov+cov.transpose(-1,-2))
            return mu, cov
        else:
            return mu, var
            

    def evaluate(self, X_test, y_test):
        data = TensorDataset(X_test.float(), y_test.float())
        loader = DataLoader(data, shuffle=False, batch_size=100)
        self.f_predictor.eval()

        true_y, mean_list, var_list = [], [], []
        for batch_id, (xi, yi) in enumerate(loader):
            xi, yi = xi.to(self.device), yi.to(self.device)
            if self.branches==1:
                mean, var = self.get_prediction_with_uncertainty(xi)
            else:
                mean, var = self.get_prediction_with_uncertainty_branched(xi)
            true_y.append(yi)
            mean_list.append(mean)
            var_list.append(var)

        true_y = torch.cat(true_y, dim=0)
        mean_list = torch.cat(mean_list, dim=0)
        var_list = torch.cat(var_list, dim=0)

        return true_y, mean_list, var_list
    

    def fantasize(self,X,sampler,observation_noise=True,**kwargs):
        propagate_grads = kwargs.pop('propagate_grads',False)
        # with fantasize_flag():
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X,observation_noise=observation_noise,**kwargs)
            Y_fantasized = sampler(post_X)
        
        # print('Fantasize X shape: {x_shape}'.format(x_shape=X.shape))
        
        
        # print('Fantasize Y shape: {y_shape}'.format(y_shape=Y_fantasized.shape))
            
        Y_fantasized = Y_fantasized.detach().clone()
        num_fantasies = Y_fantasized.shape[0]
        X_clone = X.detach().clone()
        
        if hasattr(self,"input_transform") and self.input_transform is not None:
            X_clone = self.transform_inputs(X_clone)
            X_train_orig = self.transform_inputs(self.X_train)
            y_train_orig = self.y_train.tile((X_train_orig.shape[-2]//self.X_train.shape[-2],1))
        else:
            X_train_orig = self.X_train
            y_train_orig = self.y_train
            
        # X_clone[...,-1] = X_clone[...,-1].round()
        #X_clone = X_clone.expand([num_fantasies,*X_clone.shape])
        
        # Y_mean = Y_fantasized.mean(axis=0)
        # X_train_new = X_clone.reshape(-1,self.input_dim)
        # Y_train_new = Y_fantasized.reshape(-1,num_fantasies,self.output_dim)
        # X_train = torch.cat([self.X_train,X_train_new],axis=0)
        # Y_train_prev = self.y_train.unsqueeze(1).expand(self.y_train.shape[0],num_fantasies,self.output_dim)
        # Y_train = torch.cat([Y_train_prev,Y_train_new],axis=0)
        
        X_train_new = X_clone.expand(num_fantasies,*X_clone.shape)
        X_train = torch.cat([X_train_orig.expand(*X_train_new.shape[:-2],*X_train_orig.shape[-2:]),X_train_new],axis=-2)
        Y_train = torch.cat([y_train_orig.expand(*Y_fantasized.shape[:-2],*y_train_orig.shape[-2:]),Y_fantasized],axis=-2)
                
        in_size = X_train.moveaxis(-2,0).reshape(X_train.shape[-2],-1).shape[-1]
        out_size = Y_train.moveaxis(-2,0).reshape(Y_train.shape[-2],-1).shape[-1]
        # num_fantasies = Y_fantasized.shape[0]
        # print(Y_fantasized,Y_fantasized.shape)
        # unsqueezed_Y_dim = Y_fantasized.ndim
        # if unsqueezed_Y_dim>2:
        #     for singleton_dims in range(2,unsqueezed_Y_dim):
        #         assert Y_fantasized.shape[-1]==1, 'Y_fantasized has non-singleton dimension {} that cannot be squeezed.'.format(Y_fantasized.ndim)
        #         Y_fantasized = Y_fantasized.squeeze(-1)
        # assert Y_fantasized.ndim==2, 'Y_fantasized must be dimension 2 to proceed, it is currently dimension {}.'.format(Y_fantasized.ndim)
        # X_fantasized = torch.cat([x.expand(num_fantasies,self.input_dim) for x in X_clone],axis=0)
        # Y_fantasized_stack = torch.cat([Y_fantasized[:,i:i+1] for i in range(Y_fantasized.shape[1])],axis=0)
        # X_train = torch.cat([self.X_train,X_fantasized],axis=0)
        # Y_train = torch.cat([self.y_train,Y_fantasized_stack],axis=0)
        
        config_fantasy = {key:self.config[key] for key in self.config}
        config_fantasy['n_epochs'] = 20
        
        with torch.enable_grad():
            network = Arch(self.config,'deltaenc',in_size,out_size).to(self.device)
            opt = create_optimizer(network,self.config)
            fantasy_model = DeltaEnc(network=network,
                                        config=config_fantasy,
                                        optimizer=opt,
                                        X_train=X_train,
                                        y_train=Y_train,
                                        surrogate_type=self.surrogate_type,
                                        encoding=self.encoding,
                                        target=self.target,
                                        device=self.device)
            if hasattr(self,"input_transform"):
                fantasy_model.input_transform = self.input_transform
            state_dict_prev = self.f_predictor.state_dict()
            state_dict_new = fantasy_model.f_predictor.state_dict()
            state_dict_new = {key_new:state_dict_prev[key_prev].expand(state_dict_new[key_new].shape).detach().clone() 
                              for (key_new,key_prev) in zip(state_dict_new,state_dict_prev)}
            # for key_prev,key_new in zip(state_dict_prev,state_dict_new):
            #     if state_dict_prev[key_prev].shape==state_dict_new[key_new].shape:
            #         state_dict_new[key_new] = state_dict_prev[key_prev]
            #     else:
            #         if len(state_dict_new[key_new].shape)==1:
            #             print('Previous shape: {}'.format(state_dict_prev[key_prev].shape))
            #             print('New shape: {}'.format(state_dict_new[key_new].shape))
            #             state_dict_new[key_new] = torch.tile(state_dict_prev[key_prev],(num_fantasies,))
            #             print('New shape after replacement: {}'.format(state_dict_new[key_new].shape))
                        
            #         else:
            #             print('Previous shape: {}'.format(state_dict_prev[key_prev].shape))
            #             print('New shape: {}'.format(state_dict_new[key_new].shape))
            #             state_dict_new[key_new] = torch.tile(state_dict_prev[key_prev],(num_fantasies,1))
            #             print('New shape after replacement: {}'.format(state_dict_new[key_new].shape))
            # key_final_weights = [key for key in state_dict_prev.keys() if key.endswith('linear.weight')][-1]
            # key_prefix = key_final_weights[:-13]
            # keys_final_others = [key for key in state_dict_prev.keys() if key.startswith(key_prefix) and key!=key_final_weights]
            # for key in keys_final_others:
            #     state_dict_prev[key] = torch.tile(state_dict_prev[key],(num_fantasies,))
            # state_dict_prev[key_final_weights] = torch.tile(state_dict_prev[key_final_weights],(num_fantasies,1))
            fantasy_model.f_predictor.load_state_dict(state_dict_new)
            fantasy_model.f_predictor.B = self.f_predictor.B.tile((1,fantasy_model.f_predictor.B.shape[1]//self.f_predictor.B.shape[1]))
            # print('Fantasy model training input and output data:')
            # print(fantasy_model.X_train,fantasy_model.y_train)
            # print(fantasy_model.X_train.shape,fantasy_model.y_train.shape)
            fantasy_model.fit()
            fantasy_model.eval()
            fantasy_model.is_fantasy_model = True
        return fantasy_model

class DeltaEncMF(BaseModel):
    def __init__(self,
                 network,
                 config,
                 optimizer,
                 X_train,
                 y_train,
                 surrogate_type,
                 encoding='default',
                 target='dy',  # 'y', 'dy'
                 device='cpu',
                 scale=1
                 ):

        super().__init__()
        if isinstance(network,list):
            self.multi_network = True
            self.n_epochs = [cf['n_epochs'] for cf in config]
            self.actual_batch_size = [min(cf['batch_size'], len(X_train)) for cf in config]
        else:
            self.multi_network = False
            self.n_epochs = config['n_epochs']
            self.actual_batch_size = min(config['batch_size'], len(X_train))
        
        self.f_predictor = network
        self.f_optimizer = optimizer
        self.config = config
        
        X_train = X_train.float()
        y_train = y_train.float()

        self.X_train = X_train
        self.y_train = y_train
        
        self.input_dim = X_train.shape[-1]
        self.output_dim = y_train.shape[-1]
                
        self.is_fantasy_model = False
        
        batch_shape = X_train.shape[:-2]
        self._batch_shape = batch_shape
        self.n_train = X_train.shape[-2]
        
        self.train_fid_locs = [X_train[...,-1]==i for i in X_train[...,-1].unique()]
        y_train_by_fid = [y_train[fid_loc] for fid_loc in self.train_fid_locs]
        
        # self.y_max = [Y.max(dim=0)[0] for Y in y_train_by_fid]
        # self.y_min = [Y.min(dim=0)[0] for Y in y_train_by_fid]
        self.y_max = [Y.max().detach() for Y in y_train_by_fid]
        self.y_min = [Y.min().detach() for Y in y_train_by_fid]
        self.out_scaler = lambda y,y_min,y_max:(y-y_min)/(y_max-y_min)*scale
        self.out_scaler_inv = lambda y,y_min,y_max: y*(y_max-y_min)/scale+y_min
        
        self.X_train_scaled = X_train.clone()
        self.y_train_scaled = y_train.clone()
        
        for fid_loc,y_min,y_max in zip(self.train_fid_locs,self.y_min,self.y_max):
            self.y_train_scaled[fid_loc] = self.out_scaler(self.y_train_scaled[fid_loc],y_min,y_max)
                
        self.X_train_nn = self.X_train_scaled.moveaxis(-2,0).reshape(self.n_train,-1)
        self.y_train_nn = self.y_train_scaled.moveaxis(-2,0).reshape(self.n_train,-1)
        
        self.nn_input_dim = self.X_train_nn.shape[1]
        self.nn_output_dim = self.y_train_nn.shape[1]
        
        # self.anchor_X = [X_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        # self.anchor_Y = [y_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        # self.y_max = [Y.max(axis=0)[0] for Y in self.anchor_Y]
        # self.y_min = [Y.min(axis=0)[0] for Y in self.anchor_Y]
        # self.out_scaler = lambda y,y_min,y_max:(y-y_min)/(y_max-y_min)*scale
        # self.out_scaler_inv = lambda y,y_min,y_max: y*(y_max-y_min)/scale+y_min
        
        # self.X_train = torch.cat(self.anchor_X,axis=0)
        # self.anchor_Y = [self.out_scaler(y,y_min,y_max) for (y,y_min,y_max) in zip(self.anchor_Y,self.y_min,self.y_max)]
        # self.y_train = torch.cat(self.anchor_Y,axis=0)

        self.loss_fn = nn.MSELoss()

        self.device = device
        self.surrogate_type = surrogate_type
        self.target = target
        self.encoding = encoding

        # define this member variable or botorch will not 
        # allow transformations to be applied to inputs
        #self.batch_shape = batch_shape
        # if batch_shape>1:
        #     self.train_inputs = self.X_train.expand(batch_shape,*self.X_train.shape)
        # else:
        self.train_inputs = self.X_train
        if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
            self.train_inputs = (self.train_inputs,)
            # assert self.train_inputs.shape[0]%batch_shape==0, 'Number of training samples ({}) must be divisible by batch_shape ({})'.format(self.train_inputs.shape[0],batch_shape)
            # samples_in_batch = self.train_inputs.shape[0]//batch_shape
            # self.train_inputs = tuple(self.train_inputs[i*samples_in_batch:(i+1)*samples_in_batch] for i in range(batch_shape))
            
        
    @property
    def batch_shape(self):
        return self._batch_shape
    
    @batch_shape.setter
    def batch_shape(self,value):
        self._batch_shape = value
        
    # coped from model.py
    # Originally, this method accessed the first dim of self.train_inputs (i.e self.train_inputs[0])
    # because the inputs are assumed to be in batches.
    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    # copied from exact_gp.py
    # This needs to be defined if you are passing in an input transformation.
    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
            if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
                self.train_inputs = (self.train_inputs,)            
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def fit(self):
        
        if self.multi_network:
            pass
        else:
            data = TensorDataset(self.X_train_nn, self.y_train_nn)
            loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
            self.f_predictor.train()
            # pbar = tqdm(range(self.n_epochs), desc='train epochs', leave=False)
            for epoch in range(self.n_epochs):
                avg_loss = 0.0

                for batch_id, (xi, yi) in enumerate(loader):
                    xi = xi.to(self.device)
                    yi = yi.to(self.device)
                    
                    # xi_fid = [xi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]
                    # yi_fid = [yi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]

                    if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
                        xi = self.f_predictor.input_mapping(xi)
                    #     xi_fid = [self.f_predictor.input_mapping(xi) for xi in xi_fid]
                    #     X_anchors = [self.f_predictor.input_mapping(anchor) for anchor in self.anchor_X]
                    # else:
                    #     X_anchors = [anchor for anchor in self.anchor_X]
                            
                    flipped_x = torch.flip(xi, [0])
                    if self.encoding == 'default':
                        diff_x = xi-flipped_x
                        inp = torch.cat([flipped_x, diff_x], axis=1)

                    if self.target == 'y':
                        out = yi
                    else:
                        flipped_y = torch.flip(yi, [0])
                        if self.encoding == 'default':
                            diff_y = yi-flipped_y
                            out = diff_y

                    out_hat = self.f_predictor(inp)
                    self.f_optimizer.zero_grad()
                    f_loss = self.loss_fn(out_hat.float(), out.float())
                    f_loss.backward()
                    self.f_optimizer.step()
                    avg_loss += f_loss.item()/len(loader)
                # if epoch % 25 == 0:
                #     pbar.set_description('Error %f' % avg_loss)
                # print('Epoch {} Avg.train loss {}'.format(epoch, avg_loss))
                

    def save_ckpt(self, path, name):
        state = {'epoch': self.n_epochs}
        state['state_dict'] = self.f_predictor.state_dict()
        state['B'] = self.f_predictor.B
        state['opt_state_dict'] = self.f_optimizer.state_dict()
        filename = path + '/' + name + '.ckpt'
        torch.save(state, filename)
        print('Saved Ckpts')

    def load_ckpt(self, path, name):
        saved_state = torch.load(os.path.join(
            path, name + '.ckpt'), map_location=self.device)
        self.f_predictor.load_state_dict(saved_state['state_dict'])
        self.f_predictor.B = saved_state['B']

    def _map_delta_model(self, ref, query):
        if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
            ref = self.f_predictor.input_mapping(ref)
            query = self.f_predictor.input_mapping(query)
        if self.encoding == 'default':
            diff = query - ref
        samps = torch.cat([ref, diff], 1)
        pred = self.f_predictor(samps)

        return pred

    def get_prediction_with_uncertainty(self, q,get_cov=False,original_scale=True):
        orig_input_shape = q.shape
        assert q.shape[-1]==self.input_dim, 'Expected tensor to have size=input_dim ({}) in last dimension, found tensor of shape {}'.format(self.input_dim,q.shape)        
        if q.shape[-len(self.batch_shape)-2:-2]!=self.batch_shape:
            try:
                print('Need to expand input shape {shape} to match training batch shape {batch_shape}.'.format(shape=orig_input_shape,batch_shape=self.batch_shape))
                if q.shape[-len(self.batch_shape)-2:-2]==torch.Size(len(self.batch_shape)*[1]):
                    q = q.expand(*q.shape[:-len(self.batch_shape)-2],*self.batch_shape,*q.shape[-2:])
                else:
                    q = q.expand(*self.batch_shape,*q.shape)
                    for _ in range(len(self.batch_shape)):
                        q = q.moveaxis(0,-3)
            except Exception as e:
                print('Could not make tensor of shape {shape} match training batch shape {batch_shape}.'.format(shape=orig_input_shape,batch_shape=self.batch_shape))
                print(e)
        assert q.shape[-len(self.batch_shape)-2:-2]==self.batch_shape, 'Expected tensor to have batch shape matching training batch shape {batch_shape}, instead found tensor of shape {shape}.'.format(batch_shape=self.batch_shape,shape=q.shape)
        input_shape = q.shape
        #n_test = q.shape[-2]
        test_fid_locs = [q[...,-1]==i for i in self.X_train[...,-1].unique()]
        
        q_move = q.moveaxis(-2,0)
        samples_shape = q_move.shape[:-len(self.batch_shape)-1]
        
        n_ref = 100
        # ref_choice = np.random.choice(self.X_train_nn.shape[0],n_ref*int(np.prod(samples_shape[1:])))
        ref_choice = torch.randint(self.X_train_nn.shape[0],(n_ref*int(np.prod(samples_shape[1:])),))
        ref = self.X_train_nn[ref_choice].reshape(n_ref,1,*samples_shape[1:],self.X_train_nn.shape[-1]).expand(n_ref,*samples_shape,self.X_train_nn.shape[-1]).reshape(-1,self.X_train_nn.shape[-1])
        ref_y = self.y_train_nn[ref_choice].reshape(n_ref,1,*samples_shape[1:],self.y_train_nn.shape[-1]).expand(n_ref,*samples_shape,self.y_train_nn.shape[-1]).reshape(-1,self.y_train_nn.shape[-1])
        
        
        q_combine_samples = q_move.expand(n_ref,*q_move.shape).reshape(-1,*self.batch_shape,self.input_dim)
        q_reshape = q_combine_samples.reshape(q_combine_samples.shape[0],-1)
        # n_test = q_reshape.shape[0]
        # n_ref = 100
        # all_preds = torch.zeros(nref,*q.shape[:-1],1) if q.shape[-1]==self.input_dim else torch.zeros(nref,*q.shape[:-2],1,1)
        #all_preds = torch.zeros(n_ref,self.nn_input_dim,self.nn_output_dim)
        self.f_predictor.eval()
        # ref = self.X_train_nn[np.random.choice(self.X_train_nn.shape[0],n_ref*n_test)].to(self.device)
        # ref_y = self.y_train_nn[np.random.choice(self.y_train_nn.shape[0],n_ref*n_test)].to(self.device)
        # val = self._map_delta_model(ref,q_reshape.float().tile(dims=(n_ref,1)))
        val = self._map_delta_model(ref,q_reshape.float())
        if self.target != 'y':
            val+=ref_y
            
        val = val.reshape(n_ref,*samples_shape,*self.batch_shape,self.output_dim).moveaxis(1,-2)
        assert val.shape[1:-1]==input_shape[:-1], 'Something went wrong with reshaping.'
        # if collapse_flag:
        #     for _ in range(len(self.batch_shape)):
        #         val = val.moveaxis(-3,0).mean(axis=0)
        # assert val.shape[1:-1]==orig_input_shape[:-1], 'Something went wrong with collapsing.'
            
        if original_scale:
            all_preds = torch.zeros_like(val)
            for fid_loc,ymin,ymax in zip(test_fid_locs,self.y_min,self.y_max):
                all_preds[:,fid_loc] = self.out_scaler_inv(val[:,fid_loc],ymin,ymax)
        else:
            all_preds = val
        
        mu = all_preds.mean(axis=0)
        var = all_preds.var(axis=0)
        if get_cov:
            assert self.output_dim==1, 'Output must be 1-dimensional to compute covariances.'
            all_preds = all_preds.squeeze(-1)
            mu = mu.squeeze(-1)
            var = var.squeeze(-1)
            del_pred = all_preds - all_preds.mean(axis=0)
            cov = torch.einsum('i...A,i...B->...AB',del_pred,del_pred)/(n_ref-1)
            cov = 0.5*(cov+cov.transpose(-1,-2))
            return mu, cov
        else:
            return mu, var
            

    def evaluate(self, X_test, y_test):
        data = TensorDataset(X_test.float(), y_test.float())
        loader = DataLoader(data, shuffle=False, batch_size=100)
        self.f_predictor.eval()

        true_y, mean_list, var_list = [], [], []
        for batch_id, (xi, yi) in enumerate(loader):
            xi, yi = xi.to(self.device), yi.to(self.device)
            if self.branches==1:
                mean, var = self.get_prediction_with_uncertainty(xi)
            else:
                mean, var = self.get_prediction_with_uncertainty_branched(xi)
            true_y.append(yi)
            mean_list.append(mean)
            var_list.append(var)

        true_y = torch.cat(true_y, dim=0)
        mean_list = torch.cat(mean_list, dim=0)
        var_list = torch.cat(var_list, dim=0)

        return true_y, mean_list, var_list
    

    def fantasize(self,X,sampler,observation_noise=True,**kwargs):
        propagate_grads = kwargs.pop('propagate_grads',False)
        # with fantasize_flag():
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X,observation_noise=observation_noise,**kwargs)
            Y_fantasized = sampler(post_X)
        
        # print('Fantasize X shape: {x_shape}'.format(x_shape=X.shape))
        
        
        # print('Fantasize Y shape: {y_shape}'.format(y_shape=Y_fantasized.shape))
            
        Y_fantasized = Y_fantasized.detach().clone()
        num_fantasies = Y_fantasized.shape[0]
        X_clone = X.detach().clone()
        X_clone[...,-1] = X_clone[...,-1].round()
        
        if hasattr(self,"input_transform") and self.input_transform is not None:
            X_clone = self.transform_inputs(X_clone)
            X_train_orig = self.transform_inputs(self.X_train)
            y_train_orig = self.y_train.tile((X_train_orig.shape[-2]//self.X_train.shape[-2],1))
        else:
            X_train_orig = self.X_train
            y_train_orig = self.y_train
        #X_clone = X_clone.expand([num_fantasies,*X_clone.shape])
        
        # Y_mean = Y_fantasized.mean(axis=0)
        # X_train_new = X_clone.reshape(-1,self.input_dim)
        # Y_train_new = Y_fantasized.reshape(-1,num_fantasies,self.output_dim)
        # X_train = torch.cat([self.X_train,X_train_new],axis=0)
        # Y_train_prev = self.y_train.unsqueeze(1).expand(self.y_train.shape[0],num_fantasies,self.output_dim)
        # Y_train = torch.cat([Y_train_prev,Y_train_new],axis=0)
        
        X_train_new = X_clone.expand(num_fantasies,*X_clone.shape)
        X_train = torch.cat([X_train_orig.expand(*X_train_new.shape[:-2],*X_train_orig.shape[-2:]),X_train_new],axis=-2)
        Y_train = torch.cat([y_train_orig.expand(*Y_fantasized.shape[:-2],*y_train_orig.shape[-2:]),Y_fantasized],axis=-2)
        
        in_size = X_train.moveaxis(-2,0).reshape(X_train.shape[-2],-1).shape[-1]
        out_size = Y_train.moveaxis(-2,0).reshape(Y_train.shape[-2],-1).shape[-1]
        # num_fantasies = Y_fantasized.shape[0]
        # print(Y_fantasized,Y_fantasized.shape)
        # unsqueezed_Y_dim = Y_fantasized.ndim
        # if unsqueezed_Y_dim>2:
        #     for singleton_dims in range(2,unsqueezed_Y_dim):
        #         assert Y_fantasized.shape[-1]==1, 'Y_fantasized has non-singleton dimension {} that cannot be squeezed.'.format(Y_fantasized.ndim)
        #         Y_fantasized = Y_fantasized.squeeze(-1)
        # assert Y_fantasized.ndim==2, 'Y_fantasized must be dimension 2 to proceed, it is currently dimension {}.'.format(Y_fantasized.ndim)
        # X_fantasized = torch.cat([x.expand(num_fantasies,self.input_dim) for x in X_clone],axis=0)
        # Y_fantasized_stack = torch.cat([Y_fantasized[:,i:i+1] for i in range(Y_fantasized.shape[1])],axis=0)
        # X_train = torch.cat([self.X_train,X_fantasized],axis=0)
        # Y_train = torch.cat([self.y_train,Y_fantasized_stack],axis=0)
        
        config_fantasy = {key:self.config[key] for key in self.config}
        config_fantasy['n_epochs'] = 20
        
        with torch.enable_grad():
            network = Arch(self.config,'deltaenc',in_size,out_size).to(self.device)
            opt = create_optimizer(network,self.config)
            fantasy_model = DeltaEncMF(network=network,
                                        config=config_fantasy,
                                        optimizer=opt,
                                        X_train=X_train,
                                        y_train=Y_train,
                                        surrogate_type=self.surrogate_type,
                                        encoding=self.encoding,
                                        target=self.target,
                                        device=self.device)
            if hasattr(self,"input_transform"):
                fantasy_model.input_transform = self.input_transform
            state_dict_prev = self.f_predictor.state_dict()
            state_dict_new = fantasy_model.f_predictor.state_dict()
            state_dict_new = {key_new:state_dict_prev[key_prev].expand(state_dict_new[key_new].shape).detach().clone() 
                              for (key_new,key_prev) in zip(state_dict_new,state_dict_prev)}
            # for key_prev,key_new in zip(state_dict_prev,state_dict_new):
            #     if state_dict_prev[key_prev].shape==state_dict_new[key_new].shape:
            #         state_dict_new[key_new] = state_dict_prev[key_prev]
            #     else:
            #         if len(state_dict_new[key_new].shape)==1:
            #             print('Previous shape: {}'.format(state_dict_prev[key_prev].shape))
            #             print('New shape: {}'.format(state_dict_new[key_new].shape))
            #             state_dict_new[key_new] = torch.tile(state_dict_prev[key_prev],(num_fantasies,))
            #             print('New shape after replacement: {}'.format(state_dict_new[key_new].shape))
                        
            #         else:
            #             print('Previous shape: {}'.format(state_dict_prev[key_prev].shape))
            #             print('New shape: {}'.format(state_dict_new[key_new].shape))
            #             state_dict_new[key_new] = torch.tile(state_dict_prev[key_prev],(num_fantasies,1))
            #             print('New shape after replacement: {}'.format(state_dict_new[key_new].shape))
            # key_final_weights = [key for key in state_dict_prev.keys() if key.endswith('linear.weight')][-1]
            # key_prefix = key_final_weights[:-13]
            # keys_final_others = [key for key in state_dict_prev.keys() if key.startswith(key_prefix) and key!=key_final_weights]
            # for key in keys_final_others:
            #     state_dict_prev[key] = torch.tile(state_dict_prev[key],(num_fantasies,))
            # state_dict_prev[key_final_weights] = torch.tile(state_dict_prev[key_final_weights],(num_fantasies,1))
            fantasy_model.f_predictor.load_state_dict(state_dict_new)
            fantasy_model.f_predictor.B = self.f_predictor.B.tile((1,fantasy_model.f_predictor.B.shape[1]//self.f_predictor.B.shape[1]))
            # print('Fantasy model training input and output data:')
            # print(fantasy_model.X_train,fantasy_model.y_train)
            # print(fantasy_model.X_train.shape,fantasy_model.y_train.shape)
            fantasy_model.fit()
            fantasy_model.eval()
            fantasy_model.is_fantasy_model = True
        return fantasy_model

class DeltaEncMFOrig(BaseModel):
    def __init__(self,
                 network,
                 config,
                 optimizer,
                 X_train,
                 y_train,
                 surrogate_type,
                 encoding='default',
                 target='dy',  # 'y', 'dy'
                 device='cpu',
                 batch_shape=1,
                 scale=1
                 ):

        super().__init__()
        if isinstance(network,list):
            self.multi_network = True
            self.n_epochs = [cf['n_epochs'] for cf in config]
            self.actual_batch_size = [min(cf['batch_size'], len(X_train)) for cf in config]
        else:
            self.multi_network = False
            self.n_epochs = config['n_epochs']
            self.actual_batch_size = min(config['batch_size'], len(X_train))
        
        self.f_predictor = network
        self.f_optimizer = optimizer
        self.config = config
        
        X_train = X_train.float()
        y_train = y_train.float()

        self.X_train_raw = X_train
        self.y_train_raw = y_train
        
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]
        
        self.anchor_X = [X_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        self.anchor_Y = [y_train[X_train[:,-1]==i] for i in X_train[:,-1].unique()]
        self.y_max = [Y.max(axis=0)[0] for Y in self.anchor_Y]
        self.y_min = [Y.min(axis=0)[0] for Y in self.anchor_Y]
        self.out_scaler = lambda y,y_min,y_max:(y-y_min)/(y_max-y_min)*scale
        self.out_scaler_inv = lambda y,y_min,y_max: y*(y_max-y_min)/scale+y_min
        
        self.X_train = torch.cat(self.anchor_X,axis=0)
        self.anchor_Y = [self.out_scaler(y,y_min,y_max) for (y,y_min,y_max) in zip(self.anchor_Y,self.y_min,self.y_max)]
        self.y_train = torch.cat(self.anchor_Y,axis=0)

        self.loss_fn = nn.MSELoss()

        self.device = device
        self.surrogate_type = surrogate_type
        self.target = target
        self.encoding = encoding

        # define this member variable or botorch will not 
        # allow transformations to be applied to inputs
        #self.batch_shape = batch_shape
        self.train_inputs = self.X_train
        if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
            assert self.train_inputs.shape[0]%batch_shape==0, 'Number of training samples ({}) must be divisible by batch_shape ({})'.format(self.train_inputs.shape[0],batch_shape)
            samples_in_batch = self.train_inputs.shape[0]//batch_shape
            self.train_inputs = tuple(self.train_inputs[i*samples_in_batch:(i+1)*samples_in_batch] for i in range(batch_shape))
            
        self._batch_shape = self.train_inputs[0].shape[:-2]
        
    @property
    def batch_shape(self):
        return self._batch_shape
    
    @batch_shape.setter
    def batch_shape(self,value):
        self._batch_shape = value
        
    def scale_output(self,y):
        y_new = (y-self.y_max)/(self.y_max-self.y_min)


    # coped from model.py
    # Originally, this method accessed the first dim of self.train_inputs (i.e self.train_inputs[0])
    # because the inputs are assumed to be in batches.
    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    # copied from exact_gp.py
    # This needs to be defined if you are passing in an input transformation.
    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
            if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
                self.train_inputs = (self.train_inputs,)            
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def fit(self):
        
        if self.multi_network:
            pass
        else:
            data = TensorDataset(self.X_train, self.y_train)
            loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
            self.f_predictor.train()
            # pbar = tqdm(range(self.n_epochs), desc='train epochs', leave=False)
            for epoch in range(self.n_epochs):
                avg_loss = 0.0

                for batch_id, (xi, yi) in enumerate(loader):
                    xi = xi.to(self.device)
                    yi = yi.to(self.device)
                    
                    xi_fid = [xi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]
                    yi_fid = [yi[xi[:,-1]==i] for i in self.X_train[:,-1].unique()]

                    if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
                        xi_fid = [self.f_predictor.input_mapping(xi) for xi in xi_fid]
                        X_anchors = [self.f_predictor.input_mapping(anchor) for anchor in self.anchor_X]
                    else:
                        X_anchors = [anchor for anchor in self.anchor_X]
                            
                    anchor_loc = [np.random.choice(len(X_anchors[i]),len(xi_fid[i])) for i in range(len(xi_fid))]
                    X_anchors = [X[locs] for X,locs in zip(X_anchors,anchor_loc)]
                    diff_x = [xi-X for xi,X in zip(xi_fid,X_anchors)]
                                        
                    X_anchors_torch = torch.cat(X_anchors,axis=0)
                    diff_x_torch = torch.cat(diff_x,axis=0)
                    inp = torch.cat([X_anchors_torch,diff_x_torch],axis=1)
                    
                    if self.target == 'y':
                        out = torch.cat(yi_fid,axis=0)
                    else:
                        Y_anchors = [Y[locs] for Y,locs in zip(self.anchor_Y,anchor_loc)]
                        diff_y = [yi-Y for yi,Y in zip(yi_fid,Y_anchors)]
                        out = torch.cat(diff_y,axis=0)

                    out_hat = self.f_predictor(inp)
                    self.f_optimizer.zero_grad()
                    f_loss = self.loss_fn(out_hat.float(), out.float())
                    f_loss.backward()
                    self.f_optimizer.step()
                    avg_loss += f_loss.item()/len(loader)
                # if epoch % 25 == 0:
                #     pbar.set_description('Error %f' % avg_loss)
                # print('Epoch {} Avg.train loss {}'.format(epoch, avg_loss))
                

    def save_ckpt(self, path, name):
        state = {'epoch': self.n_epochs}
        state['state_dict'] = self.f_predictor.state_dict()
        state['B'] = self.f_predictor.B
        state['opt_state_dict'] = self.f_optimizer.state_dict()
        filename = path + '/' + name + '.ckpt'
        torch.save(state, filename)
        print('Saved Ckpts')

    def load_ckpt(self, path, name):
        saved_state = torch.load(os.path.join(
            path, name + '.ckpt'), map_location=self.device)
        self.f_predictor.load_state_dict(saved_state['state_dict'])
        self.f_predictor.B = saved_state['B']

    def _map_delta_model(self, ref, query):
        if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
            ref = self.f_predictor.input_mapping(ref)
            query = self.f_predictor.input_mapping(query)
        if self.encoding == 'default':
            diff = query - ref
        samps = torch.cat([ref, diff], 1)
        pred = self.f_predictor(samps)

        return pred

    def get_prediction_with_uncertainty(self, q,get_cov=False,original_scale=True):
        assert q.shape[-1]==self.input_dim, 'Expected tensor to have size=input_dim ({}) in second-to-last dimension, found tensor of shape {}'.format(self.input_dim,q.shape)
        q_reshape = q.reshape(-1,q.shape[-1])
        fid_locs = [q_reshape[:,-1]==i for i in self.X_train[:,-1].unique()]
        n_test = q_reshape.shape[0]
        nref = 100
        # all_preds = torch.zeros(nref,*q.shape[:-1],1) if q.shape[-1]==self.input_dim else torch.zeros(nref,*q.shape[:-2],1,1)
        all_preds = torch.zeros(nref,n_test,self.output_dim)
        self.f_predictor.eval()
        for fid, locs in enumerate(fid_locs):
            n_test_fid = q_reshape[locs].shape[0]
            ref = self.anchor_X[fid][np.random.choice(self.anchor_X[fid].shape[0],nref*n_test_fid)].to(self.device)
            ref_y = self.anchor_Y[fid][np.random.choice(self.anchor_X[fid].shape[0],nref*n_test_fid)].to(self.device)
            val = self._map_delta_model(ref,q_reshape[locs].float().tile(dims=(nref,1)))
            if self.target != 'y':
                val+=ref_y
            if original_scale:
                all_preds[:,locs,:] = self.out_scaler_inv(val,self.y_min[fid],self.y_max[fid]).reshape(
                    nref,n_test_fid,self.output_dim)
            else:
                all_preds[:,locs,:] = val.reshape(nref,n_test_fid,self.output_dim)
        all_preds = all_preds.reshape(nref,*q.shape[:-1],self.output_dim)
        mu = all_preds.mean(axis=0)
        var = all_preds.var(axis=0)
        if get_cov:
            assert self.output_dim==1, 'Output must be 1-dimensional to compute covariances.'
            all_preds = all_preds.squeeze(-1)
            mu = mu.squeeze(-1)
            var = var.squeeze(-1)
            cov_dim = all_preds.shape[-1]
            del_pred = all_preds - all_preds.mean(axis=0)
            cov_mat = torch.bmm(del_pred.reshape(-1,cov_dim).unsqueeze(-1),del_pred.reshape(-1,cov_dim).unsqueeze(-2)).reshape(*all_preds.shape[:-1],cov_dim,cov_dim)
            cov = cov_mat.sum(axis=0)/(cov_mat.shape[0]-1)
            return mu, cov
        else:
            return mu, var
            
            
            

        # out = super().get_prediction_with_uncertainty(q)
        # if out is None:
        #     if len(q.shape) > 2:
        #         q = q.to(self.device).squeeze(1)
        #     fid_locs = [q[:,-1]==i for i in self.X_train[:,-1].unique()]
        #     n_test = q.shape[0]
        #     self.f_predictor.eval()
        #     mu = torch.zeros(n_test,self.output_dim).squeeze(1)
        #     var = torch.zeros(n_test,self.output_dim).squeeze(1)
        #     for fid, locs in enumerate(fid_locs):
        #         n_test_fid = q[locs].shape[0]
        #         nref = np.minimum(20,self.anchor_X[fid].shape[0])
        #         all_preds = []
        #         for i in np.random.choice(self.anchor_X[fid].shape[0],nref):
        #             ref = self.anchor_X[fid][i].to(self.device)
        #             ref_y = self.anchor_Y[fid][i].to(self.device)
        #             val = self._map_delta_model(ref.expand([n_test_fid,ref.shape[0]]),q[locs].float())
        #             if self.target != 'y':
        #                 val+=ref_y
        #             all_preds.append(self.out_scaler_inv(val,self.y_min[fid],self.y_max[fid]))
        #         all_preds = torch.stack(all_preds).squeeze(2)
        #         mu[locs] = torch.mean(all_preds,axis=0)
        #         var[locs] = torch.var(all_preds,axis=0)

        #     return mu, var
        # return out

    def evaluate(self, X_test, y_test):
        data = TensorDataset(X_test, y_test)
        loader = DataLoader(data, shuffle=False, batch_size=100)
        self.f_predictor.eval()

        true_y, mean_list, var_list = [], [], []
        for batch_id, (xi, yi) in enumerate(loader):
            xi, yi = xi.to(self.device), yi.to(self.device)
            mean, var = self.get_prediction_with_uncertainty(xi)
            true_y.append(yi)
            mean_list.append(mean)
            var_list.append(var)

        true_y = torch.cat(true_y, dim=0)
        mean_list = torch.cat(mean_list, dim=0)
        var_list = torch.cat(var_list, dim=0)

        return true_y, mean_list, var_list
    

    def fantasize(self,X,sampler,observation_noise=True,**kwargs):
        propagate_grads = kwargs.pop('propagate_grads',False)
        # with fantasize_flag():
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X,observation_noise=observation_noise,**kwargs)
            Y_fantasized = sampler(post_X)
            
        Y_fantasized = Y_fantasized.detach().clone()
        num_fantasies = Y_fantasized.shape[0]
        X_clone = X.detach().clone()
        X_clone[:,-1] = X_clone[:,-1].round()
        X_clone = X_clone.expand([num_fantasies,*X_clone.shape])
        
        # Y_mean = Y_fantasized.mean(axis=0)
        X_train_new = X_clone.reshape(-1,self.input_dim)
        Y_train_new = Y_fantasized.reshape(-1,self.output_dim)
        X_train = torch.cat([self.X_train,X_train_new],axis=0)
        Y_train = torch.cat([self.y_train,Y_train_new],axis=0)
        
        # num_fantasies = Y_fantasized.shape[0]
        # print(Y_fantasized,Y_fantasized.shape)
        # unsqueezed_Y_dim = Y_fantasized.ndim
        # if unsqueezed_Y_dim>2:
        #     for singleton_dims in range(2,unsqueezed_Y_dim):
        #         assert Y_fantasized.shape[-1]==1, 'Y_fantasized has non-singleton dimension {} that cannot be squeezed.'.format(Y_fantasized.ndim)
        #         Y_fantasized = Y_fantasized.squeeze(-1)
        # assert Y_fantasized.ndim==2, 'Y_fantasized must be dimension 2 to proceed, it is currently dimension {}.'.format(Y_fantasized.ndim)
        # X_fantasized = torch.cat([x.expand(num_fantasies,self.input_dim) for x in X_clone],axis=0)
        # Y_fantasized_stack = torch.cat([Y_fantasized[:,i:i+1] for i in range(Y_fantasized.shape[1])],axis=0)
        # X_train = torch.cat([self.X_train,X_fantasized],axis=0)
        # Y_train = torch.cat([self.y_train,Y_fantasized_stack],axis=0)
        
        config_fantasy = {key:self.config[key] for key in self.config}
        config_fantasy['n_epochs'] = 20
        
        with torch.enable_grad():
            network = Arch(self.config,'deltaenc',self.input_dim,self.output_dim).to(self.device)
            opt = create_optimizer(network,self.config)
            fantasy_model = DeltaEncMF(network=network,
                                        config=config_fantasy,
                                        optimizer=opt,
                                        X_train=X_train_new,
                                        y_train=Y_train_new,
                                        surrogate_type=self.surrogate_type,
                                        encoding=self.encoding,
                                        target=self.target,
                                        device=self.device,
                                        batch_shape=1)
            fantasy_model.f_predictor.load_state_dict(self.f_predictor.state_dict())
            # print('Fantasy model training input and output data:')
            # print(fantasy_model.X_train,fantasy_model.y_train)
            # print(fantasy_model.X_train.shape,fantasy_model.y_train.shape)
            fantasy_model.fit()
            fantasy_model.eval()
        return fantasy_model



class DeltaEncOrig(BaseModel):
    def __init__(self,
                 network,
                 config,
                 optimizer,
                 X_train,
                 y_train,
                 surrogate_type,
                 encoding='default',
                 target='dy',  # 'y', 'dy'
                 device='cpu',
                 batch_shape=1
                 ):

        super().__init__()
        self.f_predictor = network
        self.f_optimizer = optimizer
        self.config = config

        self.X_train = X_train.float()
        self.y_train = y_train.float()
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]

        self.loss_fn = nn.MSELoss()
        self.n_epochs = config['n_epochs']
        self.actual_batch_size = min(config['batch_size'], len(self.X_train))

        self.device = device
        self.surrogate_type = surrogate_type
        self.target = target
        self.encoding = encoding

        # define this member variable or botorch will not 
        # allow transformations to be applied to inputs
        #self.batch_shape = batch_shape
        self.train_inputs = self.X_train
        if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
            assert self.train_inputs.shape[0]%batch_shape==0, 'Number of training samples ({}) must be divisible by batch_shape ({})'.format(self.train_inputs.shape[0],batch_shape)
            samples_in_batch = self.train_inputs.shape[0]//batch_shape
            self.train_inputs = tuple(self.train_inputs[i*samples_in_batch:(i+1)*samples_in_batch] for i in range(batch_shape))
            
        self._batch_shape = self.train_inputs[0].shape[:-2]
        
    @property
    def batch_shape(self):
        return self._batch_shape
    
    @batch_shape.setter
    def batch_shape(self,value):
        self._batch_shape = value

    # coped from model.py
    # Originally, this method accessed the first dim of self.train_inputs (i.e self.train_inputs[0])
    # because the inputs are assumed to be in batches.
    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    # copied from exact_gp.py
    # This needs to be defined if you are passing in an input transformation.
    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
            if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
                self.train_inputs = (self.train_inputs,)            
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def fit(self):
        data = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)

        self.f_predictor.train()
        # pbar = tqdm(range(self.n_epochs), desc='train epochs', leave=False)
        for epoch in range(self.n_epochs):
            avg_loss = 0.0

            for batch_id, (xi, yi) in enumerate(loader):
                xi = xi.to(self.device)
                yi = yi.to(self.device)

                if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
                    xi = self.f_predictor.input_mapping(xi)

                flipped_x = torch.flip(xi, [0])
                if self.encoding == 'default':
                    diff_x = xi-flipped_x
                    inp = torch.cat([flipped_x, diff_x], axis=1)

                if self.target == 'y':
                    out = yi
                else:
                    flipped_y = torch.flip(yi, [0])
                    if self.encoding == 'default':
                        diff_y = yi-flipped_y
                        out = diff_y

                out_hat = self.f_predictor(inp)
                self.f_optimizer.zero_grad()
                f_loss = self.loss_fn(out_hat.float(), out.float())
                f_loss.backward()
                self.f_optimizer.step()
                avg_loss += f_loss.item()/len(loader)
            # if epoch % 25 == 0:
            #     pbar.set_description('Error %f' % avg_loss)
            # print('Epoch {} Avg.train loss {}'.format(epoch, avg_loss))

    def save_ckpt(self, path, name):
        state = {'epoch': self.n_epochs}
        state['state_dict'] = self.f_predictor.state_dict()
        state['B'] = self.f_predictor.B
        state['opt_state_dict'] = self.f_optimizer.state_dict()
        filename = path + '/' + name + '.ckpt'
        torch.save(state, filename)
        print('Saved Ckpts')

    def load_ckpt(self, path, name):
        saved_state = torch.load(os.path.join(
            path, name + '.ckpt'), map_location=self.device)
        self.f_predictor.load_state_dict(saved_state['state_dict'])
        self.f_predictor.B = saved_state['B']

    def _map_delta_model(self, ref, query):
        if self.surrogate_type == 'FF' or self.surrogate_type == 'SIREN_FF':
            ref = self.f_predictor.input_mapping(ref)
            query = self.f_predictor.input_mapping(query)
        if self.encoding == 'default':
            diff = query - ref
        samps = torch.cat([ref, diff], 1)
        pred = self.f_predictor(samps)

        return pred

    def get_prediction_with_uncertainty(self, q):
        out = super().get_prediction_with_uncertainty(q)
        if out is None:
            if len(q.shape) > 2:
                q = q.to(self.device).squeeze(1)
            nref = np.minimum(20, self.X_train.shape[0])
            self.f_predictor.eval()
            all_preds = []
            n_test = q.shape[0]
            for i in list(np.random.choice(self.X_train.shape[0], nref)):
                ref = self.X_train[i].to(self.device)
                ref_y = self.y_train[i].to(self.device)
                if self.target == 'y':
                    val = self._map_delta_model(ref.expand([n_test, ref.shape[0]]), q.float())
                else:
                    val = self._map_delta_model(ref.expand([n_test, ref.shape[0]]), q.float()) + ref_y
                all_preds.append(val)

            all_preds = torch.stack(all_preds).squeeze(2)
            mu = torch.mean(all_preds, axis=0)
            var = torch.var(all_preds, axis=0)

            return mu, var
        return out

    def evaluate(self, X_test, y_test):
        data = TensorDataset(X_test.float(), y_test.float())
        loader = DataLoader(data, shuffle=False, batch_size=100)
        self.f_predictor.eval()

        true_y, mean_list, var_list = [], [], []
        for batch_id, (xi, yi) in enumerate(loader):
            xi, yi = xi.to(self.device), yi.to(self.device)
            mean, var = self.get_prediction_with_uncertainty(xi)
            true_y.append(yi)
            mean_list.append(mean)
            var_list.append(var)

        true_y = torch.cat(true_y, dim=0)
        mean_list = torch.cat(mean_list, dim=0)
        var_list = torch.cat(var_list, dim=0)

        return true_y, mean_list, var_list
    

    def fantasize(self,X,sampler,observation_noise=True,**kwargs):
        propagate_grads = kwargs.pop('propagate_grads',False)
        # with fantasize_flag():
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X.float(),observation_noise=observation_noise,**kwargs)
            Y_fantasized = sampler(post_X)
            
        Y_fantasized = Y_fantasized.detach().clone().float()
        num_fantasies = Y_fantasized.shape[0]
        X_clone = X.detach().clone().float()
        X_clone[:,-1] = X_clone[:,-1].round()
        X_clone = X_clone.expand([num_fantasies,*X_clone.shape])
        
        # Y_mean = Y_fantasized.mean(axis=0)
        X_train_new = X_clone.reshape(-1,self.input_dim)
        Y_train_new = Y_fantasized.reshape(-1,self.output_dim)
        X_train = torch.cat([self.X_train,X_train_new],axis=0)
        Y_train = torch.cat([self.y_train,Y_train_new],axis=0)
        
        # num_fantasies = Y_fantasized.shape[0]
        # print(Y_fantasized,Y_fantasized.shape)
        # unsqueezed_Y_dim = Y_fantasized.ndim
        # if unsqueezed_Y_dim>2:
        #     for singleton_dims in range(2,unsqueezed_Y_dim):
        #         assert Y_fantasized.shape[-1]==1, 'Y_fantasized has non-singleton dimension {} that cannot be squeezed.'.format(Y_fantasized.ndim)
        #         Y_fantasized = Y_fantasized.squeeze(-1)
        # assert Y_fantasized.ndim==2, 'Y_fantasized must be dimension 2 to proceed, it is currently dimension {}.'.format(Y_fantasized.ndim)
        # X_fantasized = torch.cat([x.expand(num_fantasies,self.input_dim) for x in X_clone],axis=0)
        # Y_fantasized_stack = torch.cat([Y_fantasized[:,i:i+1] for i in range(Y_fantasized.shape[1])],axis=0)
        # X_train = torch.cat([self.X_train,X_fantasized],axis=0)
        # Y_train = torch.cat([self.y_train,Y_fantasized_stack],axis=0)
        config_fantasy = {key:self.config[key] for key in self.config}
        config_fantasy['n_epochs'] = 20
        
        with torch.enable_grad():
            network = Arch(self.config,'deltaenc',self.input_dim,self.output_dim).to(self.device)
            opt = create_optimizer(network,self.config)
            fantasy_model = DeltaEnc(network=network,
                                        config=config_fantasy,
                                        optimizer=opt,
                                        X_train=X_train_new,
                                        y_train=Y_train_new,
                                        surrogate_type=self.surrogate_type,
                                        encoding=self.encoding,
                                        target=self.target,
                                        device=self.device,
                                        batch_shape=1)
            fantasy_model.f_predictor.load_state_dict(self.f_predictor.state_dict())
            # print('Fantasy model training input and output data:')
            # print(fantasy_model.X_train,fantasy_model.y_train)
            # print(fantasy_model.X_train.shape,fantasy_model.y_train.shape)
            fantasy_model.fit()
            fantasy_model.eval()
        return fantasy_model
