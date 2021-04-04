from helper import *
from ml_methods import *
from datetime import datetime
import sklearn
import time
import pickle as pkl

class basic_ml():
    def __init__(self):
        pass

    def learner(self, model, X, y, test_param, var_to_learn, sample_weights = None):
        if isinstance(var_to_learn, list):
            for i,v in enumerate(var_to_learn):
                setattr(model, v, test_param[i])
        elif var_to_learn is not None:
            if var_to_learn == 'C':
                setattr(model, var_to_learn, 1/test_param)
            else:
                setattr(model, var_to_learn, test_param)
        params = model.get_params()
        if 'class_weight' in params and params['class_weight'] == None:
            clf = model.fit(X, y, sample_weight = sample_weights)
        else:
            clf = model.fit(X, y)
        return clf
    
    def starter(self, model, x, targets, dtype, ttype, add_date = True):
        if model.random_state:
            seed = model.random_state
        else:
            seed = 0
        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        y = targets
        X = np.array(x)
        return seed, X, y


    def train_func(self, model, ix_in, X, y, tmpts, ts_true_in, ts_pred_in, ts_probs_in, loss_vec_in, test_param = None, var_to_learn = None):
        train_index_in, test_index_in = ix_in
        X_train_in, X_test_in = X[train_index_in, :], X[test_index_in, :]
        y_train_in, y_test_in = y[train_index_in], y[test_index_in]

        tmpts_train_in = tmpts[train_index_in]
        coefs_all_in = []
        samp_weights = get_class_weights(y_train_in, tmpts_train_in)

        clf = self.learner(model, X_train_in, y_train_in, test_param, var_to_learn, sample_weights = samp_weights) 
        
        y_probs_train = clf.predict_proba(X_train_in)
        y_guess = clf.predict(X_test_in)
        y_probs = clf.predict_proba(X_test_in)

        if len(y_test_in)>1:
            ts_true_in.extend(y_test_in.squeeze())
            ts_pred_in.extend(y_guess.squeeze())
            ts_probs_in.extend(y_probs[:,1].squeeze())
        else:
            ts_true_in.append(int(y_test_in.squeeze()))
            ts_pred_in.append(int(y_guess.squeeze()))
            ts_probs_in.append(float(y_probs[:,1].squeeze()))

        if (y_probs[:,1] == 0).any() or ((1-y_probs[:, 1])==0).any():
            try:
                loss = 1. if y_test_in.item() == y_probs[:,1].item() else 0.
            except:
                loss = np.sum(1-(y_test_in == y_probs[:, 1]).astype(int))/len(y_test_in)
        else:
            loss = (-y_test_in * np.log(y_probs[:, 1]) - (1-y_test_in)*np.log(1-y_probs[:, 1]))
        
        if not isinstance(loss, int) and not isinstance(loss, float):
            if loss.shape[0]>1:
                loss = (np.sum(loss)/loss.shape[0]).item()
            else:
                loss = loss.item()

        # import pdb; pdb.set_trace()
        loss_vec_in.append(loss)
        # assert(y_guess.item() == np.round(y_probs[:,1].item()))

        return ts_true_in, ts_pred_in, ts_probs_in, loss_vec_in, clf, y_probs_train

    def fit_all(self, model, x, targets, name='', dtype = 'metabolites', ttype = 'week_one', \
        optim_param = 'auc', var_to_learn = 'C', optimal_param = 7.753):
        seed, X, y = self.starter(model, x, targets, dtype, ttype)
        final_res_dict = {}
        tmpts = np.array([ix.split('-')[1] for ix in x.index.values])
        samp_weights = get_class_weights(y, tmpts)
        clf = self.learner(model, X, y, optimal_param, var_to_learn, sample_weights = samp_weights)
        y_guess = clf.predict(X)
        y_probs = clf.predict_proba(X)[:,1]
        ret_dict = get_metrics(y_guess, y, y_probs)
        final_res_dict['metrics'] = ret_dict
        final_res_dict['model'] = clf
        if 'coef_' in clf.get_params().keys():
            final_res_dict['coef'] = clf.coef_
        return final_res_dict



    def one_cv_func(self, model, x, targets, split_outer = leave_one_out_cv, name = '', dtype = 'metabolites', ttype = 'week_one', \
        folds = None, optim_param = 'auc', final_res_dict = {}, var_to_learn = 'C', test_param = 1):
        seed, X, y= self.starter(model, x, targets, dtype, ttype)

        final_res_dict = {}
        ixs = split_outer(x, y, folds = folds, ddtype = ttype)
        tmpts = np.array([ix.split('-')[1] for ix in x.index.values])

        ts_true = []
        ts_pred = []
        loss_vec = []
        ts_probs = []
        coefs_all = {}
        model_all = {}
        for ic, ix in enumerate(ixs):         
            
            ts_true, ts_pred, ts_probs, loss_vec, clf, y_probs_tr = self.train_func(model, ix, X, y, tmpts, \
                ts_true, ts_pred, ts_probs, loss_vec, var_to_learn = var_to_learn, test_param = test_param)
            
            try:
                coefs_all[ic] = clf.coef_
            except:
                continue
            model_all[ic] = clf

            print('split ' + str(ic) + ' complete')

        ret_dict = get_metrics(ts_pred, ts_true, ts_probs)
        # print(ts_true)
        # print(ts_pred)
        # print(ts_probs)

        final_res_dict['metrics'] = ret_dict
        if len(coefs_all) >0:
            final_res_dict['coef'] = coefs_all
        final_res_dict['model'] = model_all
        return final_res_dict


    def nested_cv_func(self, model, x, targets, feature_grid = np.logspace(-3, 3, 100), \
        split_outer = leave_one_out_cv, split_inner = leave_one_out_cv, learn_var = 'C', nzero_thresh = 10, \
            name = '', dtype = 'metabolites', ttype = 'week_one', folds = None, optim_param = 'auc', plot_lambdas = False, \
                stop_lambdas = False, smooth_auc = True):
        seed, X, y = self.starter(model, x, targets, dtype, ttype)
        final_res_dict = {}

        ixs = split_outer(x, y, folds = folds, ddtype = ttype)
        tmpts = np.array([ix.split('-')[1] for ix in x.index.values])

        ts_true = []
        ts_pred = []
        loss_vec = []
        ts_probs = []
        coefs_all = {}
        model_all = {}
        best_lambdas = []
        best_auc_vec = []
        best_auc_vec_ma = []
        training_probs = []
        train_auc_outer = []
        auc_score_tr = []
        test_auc_dict={}
        for ic, ix in enumerate(ixs):
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            x_train_to_split = x.iloc[train_index,:]
            ixs_inner = split_inner(x_train_to_split, y_train, folds = folds, ddtype = ttype) 
            lambdict = {}
            start = time.time()
            train_auc_dict = {}
            for lamb in feature_grid:
                lambdict[lamb] = {}
                train_auc_dict[lamb] = {}
                test_auc_dict[lamb] = {}


                ts_true_in = []
                ts_pred_in = []
                loss_vec_in = []
                ts_probs_in = []
                train_auc = []
                for ic_in,ix_in in enumerate(ixs_inner):
                    ts_true_in, ts_pred_in, ts_probs_in, loss_vec_in, clf, y_probs_tr = self.train_func(model, \
                        ix_in, X_train, y_train, tmpts, ts_true_in, ts_pred_in, ts_probs_in, loss_vec_in, \
                            test_param = lamb, var_to_learn = learn_var)

                    if 'coef_' in clf.get_params().keys():
                        num_coefs = sum(clf.coef_!=0)
                        if np.mean(num_coefs)< nzero_thresh and stop_lambdas:
                            del lambdict[lamb]
                            continue
                    train_auc.append(sklearn.metrics.roc_auc_score(y_train[ix_in[0]], y_probs_tr[:,1]))
                train_auc_dict[lamb][ic] = train_auc
                # print('AUC for lambda ' + str(lamb) + '= ' + str(train_auc))

                met_dict = get_metrics(ts_pred_in, ts_true_in, ts_probs_in)

                test_auc_dict[lamb][ic] = met_dict
                lambdict[lamb] = met_dict
                lambdict[lamb]['loss'] = np.sum(loss_vec_in)/len(loss_vec_in)
            end = time.time()
            # if ic == 0:
            #     print('Time for innermost loop: ' + str(end - start))
            lambdas = list(lambdict.keys())
            key = optim_param
            vec = np.array([lambdict[it][key] for it in lambdas])
            ma = moving_average(vec, n=5)
            best_param = np.max(vec)
            best_param_ma = np.max(ma)
            offset = int(np.floor(5./2))
            if key == 'loss':
                if smooth_auc:
                    max_ix  = np.argmin(ma) + offset
                else:
                    max_ix  = np.argmin(vec)
            else:
                if smooth_auc:
                    max_ix = np.argmax(ma) + offset
                else:
                    max_ix = np.argmax(vec)
            best_lambda = lambdas[max_ix]

            if plot_lambdas:
                fig2, ax2 = plot_lambdas_func(lambdict, optim_param, offset, ma, best_lambda)

            ts_true, ts_pred, ts_probs, loss_vec, clf, y_probs_tr = self.train_func(model, ix, X, y, tmpts, \
                ts_true, ts_pred, ts_probs, loss_vec, test_param = best_lambda, var_to_learn = learn_var)
            
            if 'coef_' in clf.get_params().keys():
                coefs_all[ic] = clf.coef_
            model_all[ic] = clf

            # print('split ' + str(ic) + ' complete')
            best_lambdas.append(best_lambda)
        
            best_auc_vec.append(best_param)
            best_auc_vec_ma.append(best_param_ma)

            training_probs.append((y[ix[0]], y_probs_tr))

            auc_score_tr.append(sklearn.metrics.roc_auc_score(y_train, y_probs_tr[:,1]))

        ret_dict = get_metrics(ts_pred, ts_true, ts_probs)

        final_res_dict['best_lambda'] = best_lambdas
        final_res_dict['metrics'] = ret_dict
        final_res_dict['coef'] = coefs_all
        final_res_dict['model'] = model_all
        final_res_dict['lambdict'] = test_auc_dict
        final_res_dict['train_fit'] = training_probs
        final_res_dict['train_auc'] = auc_score_tr
        final_res_dict['train_auc_inner'] = train_auc_dict
        return final_res_dict

    def double_nest(self, model, x, targets, feature_grid = np.logspace(-3, 3, 100), \
        split_outer = leave_one_out_cv, split_inner = leave_one_out_cv, learn_var = 'C', nzero_thresh = 10, \
            name = '', dtype = 'metabolites', ttype = 'week_one', folds = None, optim_param = 'auc', plot_lambdas = True):
        seed, X, y = self.starter(model, x, targets, dtype, ttype)

        ixs = split_outer(x, y, folds = folds, ddtype = ttype)
        tmpts = np.array([ix.split('-')[1] for ix in x.index.values])

        out_metrics = {}
        coef_vec = []
        model_vec = []
        final_res_dict = {}
        for ic, ix in enumerate(ixs):
            train_index, test_index = ix
            X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            start = time.time()
            res_dict = self.nested_cv_func(model, X_train, y_train, feature_grid = feature_grid, \
                split_outer = split_outer, split_inner = split_inner, learn_var = learn_var, nzero_thresh = nzero_thresh, \
                name = name, dtype = dtype, ttype = ttype, folds = folds, optim_param = optim_param, plot_lambdas = plot_lambdas)
            end = time.time()
            for metric in list(res_dict['metrics'].keys()):
                if metric in out_metrics.keys():
                    out_metrics[metric].append(res_dict['metrics'][metric])
                else:
                    out_metrics[metric] = [res_dict['metrics'][metric]]
            coef_vec.append(res_dict['coef'])
            model_vec.append(res_dict['model'])
            # print('idx ' + str(ic) + ' complete for outermost nest')
            # print('Time for middle nest ' + str(end - start))
                
        final_res_dict['metrics'] = out_metrics
        final_res_dict['coef'] = coef_vec
        final_res_dict['model'] = model_vec
        return final_res_dict

if __name__ == "__main__":
    mb = basic_ml()
    paths = ['week_one_ALL/', 'week_one_16s/','week_one_metabs/']
    in_file = 'x.pkl'
    dat_dict = {}
    for path in paths:
        with open('inputs/' + path + in_file,'rb') as f:
            x = pkl.load(f)
        with open('inputs/' + path + 'y.pkl','rb') as f:
            y = pkl.load(f)
        dat_dict[path[:-1]] = (x,y)

    x, y = dat_dict['week_one_ALL']
    model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', \
                               random_state = 0, solver = 'liblinear')
    out_dict = mb.nested_cv_func(model, x, y, dtype = 'metabolites', optim_param = 'auc')