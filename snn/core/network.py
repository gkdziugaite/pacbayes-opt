import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from snn.core.mlp_fn import *
from snn.core.extra_fn import *
from snn.core.data_fn import *
from snn.core.sgd import *
from snn.core import package_path
from snn.core import config
import gzip
import pickle


class Network(object):
    """ A network model """

    def __init__(self, X, Y, logging=True, layers=[784, 600, 10], scopes_list=['hidden1', 'output'], graph=tf.Graph(),
                 seed=11):
        # Initialize the tensorflow graph and session
        self.graph = graph
        with self.graph.as_default():
            tf.set_random_seed(seed)
            np.random.seed(seed)
            self.sess = tf.Session(graph=self.graph, config=config)

            # Storing variables
            self.X = X
            self.Y = Y
            self.layers = layers
            self.scopes_list = scopes_list
            self.Nsamples = self.X.shape[0]
            self.output_dict = {"L2": [], "diff": [], "weights": [], "mean_weights": [], "var_weights": [], "PACBound": [], "B_val": [], "KL_val": [], "test_acc": [], "train_acc": [], "L2_PACB": [], "log_post_all": [], "PACB_weights": [], "log_prior_std": []}

            # PACBound parameters
            self.log_prior_std_precision = 100.0
            self.log_prior_std_base = 0.1
            self.deltaPAC = 0.025

            # Set graph level random seeds
            self.seed = seed
        return

    def __call__(self, x, y):
        # Runs the accuracy of model on X
        self.print_accuracy(x, y)
        return

    def __del__(self):
        return

    @lazy_property # This causes tf_init to only be called once
    def tf_init(self):
        with self.graph.as_default():
            init = tf.initialize_all_variables()
            self.sess.run(init)
        return

    def force_tf_init(self):
        with self.graph.as_default():
            init = tf.initialize_all_variables()
            self.sess.run(init)
        return

    def print_accuracy_in_batches(self,x,y, no_batches=10, whichset = 'train'):
        ntest,no_batches = x.shape[0], int(no_batches)
        testidx = np.linspace(0, ntest, no_batches+1)
        test_acc = 0
        for (ii, jj) in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            test_acc += self.sess.run(self.accuracy,
                                      feed_dict={self.x: x[ii:jj],
                                                 self.y: y[ii:jj]})
        test_acc = test_acc / no_batches
        print("Average %s accuracy: %.4f" %(whichset, test_acc))
        return test_acc

    def print_accuracy_in_batches_noise(self, x, y, noise, feed_input, accuracy, no_batches=10, whichset='train'):
        ntest, no_batches = x.shape[0], int(no_batches)
        testidx = np.linspace(0, ntest, no_batches + 1)
        test_acc = 0
        for (ii, jj) in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            output_list = [x[ii:jj], y[ii:jj]] + noise
            feed_dict = {a: b for (a, b) in zip(feed_input, output_list)}
            test_acc += accuracy.eval(feed_dict=feed_dict)
        test_acc = test_acc / no_batches
        print("Average %s accuracy: %.4f" %(whichset, test_acc))
        return test_acc

    def logistic_loss(self, yhat):
      return tf.reduce_mean(tf.log(1 + tf.exp(-self.y * yhat)) / (np.float32(np.log(2))))

    def optimize(self, epochs=10, batch_size=100, learning_rate=0.01, momentum=0.9):
        with self.graph.as_default():
            train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(self.cost_fn)
            self.tf_init # Initialize the tf variables
            weights_rand_init = self.get_model_weights()
            # Initialize the SGD trainer
            sgd = SGD(self.X, self.Y, total_epochs=epochs, batch_size=100, seed=self.seed)
            number_of_batches_per_epoch = int(self.Nsamples / batch_size)
            total_number_of_iterations = epochs * number_of_batches_per_epoch
            for i in range(total_number_of_iterations):
                epoch = int(i / number_of_batches_per_epoch)
                if i%100==0:
                  remainder = i % number_of_batches_per_epoch
                  print("Iter: %d / %d"%(remainder, number_of_batches_per_epoch))

                if i%number_of_batches_per_epoch == 0: # Print and shuffle at every epoch
                    print("Epoch: ", epoch)
                    self.print_accuracy_in_batches(self.X, self.Y,50)
                    index_set = sgd.epoch_train_set(epoch=epoch)

                batch_x, batch_y = next_batch(self.X[index_set], self.Y[index_set], batch_size,
                                              int(i % number_of_batches_per_epoch))
                _, cost = self.sess.run([train_step, self.cost_fn], feed_dict={self.x: batch_x, self.y: batch_y})
        return weights_rand_init

    def save_output(self, path=os.path.join(package_path, "experiments", "results", "out.pickle")):
        save_dir = os.path.join(*os.path.split(path)[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(path, 'wb') as f:
            pickle.dump(self.output_dict, f)
        return

    @lazy_property
    def cost_fn(self):
        with self.graph.as_default():
            num_classes = self.y.get_shape().as_list()[-1]
            print("Number of classes: ", num_classes)
            if num_classes == 1:
                print("Minimizing logistic loss")
                return self.logistic_loss(self.yhat)
            else:
                print("Minimizing sigmoid_cross_entropy_with_logits")
                return (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yhat, labels=self.y)) /
                        tf.log(tf.to_float(num_classes)))

    @lazy_property
    def accuracy(self):
        with self.graph.as_default():
            if self.y.get_shape().as_list()[-1] == 1:
                correct_prediction = tf.equal(tf.cast(self.yhat >= 0, tf.float32) - tf.cast(self.yhat < 0, tf.float32),
                                              self.y)
            else:
                correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.yhat, axis = 1),1), tf.argmax(self.y,1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def print_full_accuracy(self, x=None, y=None):
        print("Train Accuracy:", self.sess.run(self.accuracy, feed_dict={self.x: self.X, self.y: self.Y}))
        if x is not None: print("Test Accuracy:", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}))
        return

    def print_accuracy(self, x=None, y=None):
        if x is not None: print("Accuracy:", self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y}))
        return

    def return_accuracy(self, x, y):
        with self.sess as session:
            return self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})

    def save_model(self, save_tag):
        return

    def log_res(self, res, log_tag):
        """ Logs the result """
        return

    @lazy_property
    def PACB_init(self):
        """ Initializes variables for the PAC Bound optimization """
        network_weights = self.get_model_weights()

        # Hacky way to Create a new graph and session for now, resets the default:
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.sess = tf.Session(graph=self.graph, config=config)

            # Recreate Placeholders for the optimization
            self.x, self.y = self.create_placeholders()

            # The initial standard deviations
            log_post_std_init_list = []
            init_log_prior_std = -3.0
            for _w in network_weights:
              log_post_std_init = np.log(2 * np.abs(_w))
              log_post_std_init_list.append(log_post_std_init)

            self.param_noise_list = []
            for _w in network_weights:
                self.param_noise_list.append(tf.placeholder("float", _w.shape))

            log_prior_std = variable_initializer('log_prior_std', [1], tf.constant_initializer(init_log_prior_std))

            log_post_std_list = []
            for (scopename,w_init,b_init) in zip(self.scopes_list,log_post_std_init_list[0::2],log_post_std_init_list[1::2]):
                with tf.variable_scope(scopename) as scope:
                    log_post_std_list.append(variable_initializer('log_post_std_w', shape =w_init.shape,
                                                                  initializer = tf.constant_initializer(w_init)))
                    log_post_std_list.append(variable_initializer('log_post_std_b', shape =b_init.shape,
                                                                  initializer = tf.constant_initializer(b_init)))

            network_perturb_list = []
            for (log_post_std,param_noise) in zip(log_post_std_list, self.param_noise_list):
                network_perturb_list.append(tf.multiply(tf.exp(log_post_std),param_noise))

        return network_weights, network_perturb_list, log_post_std_list, log_prior_std

    def PACB_objective(self, prior_weights, trainWeights=True):

        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            # Obtain the effective number of data points
            effective_m = self.X.shape[0]

            # Gather the initial weights to create a new optimization network
            network_weights, network_perturb_list, log_post_std_list, log_prior_std = self.PACB_init

            yhat, param_var_list = self.model_with_noise(self.x, network_perturb_list, self.scopes_list, self.layers, network_weights, graph=self.graph, trainable=trainWeights)

            norm_post_variance = tf.add_n(list(map(lambda x: tf.reduce_sum(tf.exp(x*2)), log_post_std_list)))
            norm_params = tf.add_n(list(map(lambda x,y: tf.reduce_sum((x-y)**2), param_var_list, prior_weights)))
            sum_log_post_variance = tf.add_n(list(map(lambda x: tf.reduce_sum(x), log_post_std_list)))

            correct_prediction = tf.equal(tf.cast(yhat >= 0, tf.float32) - tf.cast(yhat < 0, tf.float32),
                                          self.y)
            self.accuracySUM = tf.reduce_sum(tf.cast(correct_prediction, "float"))
            A = self.logistic_loss(yhat)

            nparams, layer_shapes = self.count_N_params()
            self.layer_shapes = layer_shapes
            self.mean_weights_component = (norm_params)/(tf.exp(2*log_prior_std))
            self.var_weights_component = norm_post_variance/(tf.exp(2*log_prior_std)) - 2*sum_log_post_variance + 2*nparams*log_prior_std
            self.KLdivTimes2 = self.mean_weights_component + self.var_weights_component - nparams
            Bquad = self.KLdivTimes2/2 + tf.log(np.pi**2 * effective_m/(6*0.05)) \
                                          + 2*tf.log(self.log_prior_std_precision) \
                                          + 2*tf.log(tf.log(self.log_prior_std_base) - 2 * log_prior_std)

            c = Bquad/(2*(effective_m-1))
            self.B = tf.sqrt(c)
            cost = A + self.B
            return A, cost, Bquad, effective_m, log_prior_std, log_post_std_list

    def PACB_store(self, save_dict, i, log_prior_std=None, m_w=None, v_w=None, bpac=None, B_val=None, KL_val=None,
                   test_acc=None, train_acc=None, log_post_std_list=None):
        if save_dict is not None:
            if i % save_dict["iter"] == 0: # Only save at a certain number of iterations
                if save_dict["mean_weights"] and m_w is not None:
                    self.output_dict["mean_weights"].append(m_w)
                if save_dict["var_weights"] and v_w is not None:
                    self.output_dict["var_weights"].append(v_w)
                if save_dict["PACBound"] and bpac is not None:
                    self.output_dict["PACBound"].append(bpac) # Store the desired values for saving
                if save_dict["B_val"] and B_val is not None:
                    self.output_dict["B_val"].append(B_val) # Store the desired values for saving
                if save_dict["KL_val"] and KL_val is not None:
                    self.output_dict["KL_val"].append(KL_val) # Store the desired values for saving
                if save_dict["test_acc"] and test_acc is not None:
                    self.output_dict["test_acc"].append(test_acc) # Store the desired values for saving
                if save_dict["train_acc"] and train_acc is not None:
                    self.output_dict["train_acc"].append(train_acc) # Store the desired values for saving
                if save_dict["L2_PACB"]:
                    self.output_dict["L2_PACB"].append(l2_norm(self.get_model_weights(), save_dict["w*"]))
                if save_dict["log_post_all"] and log_post_std_list is not None:
                    noisevecs = []
                    for log_post_std in log_post_std_list:
                        noisevecs.append(log_post_std.eval(session=self.sess))
                    log_post_std_all = []
                    for nv in noisevecs:
                        for v in nv.flatten():
                              log_post_std_all.append(v)
                    self.output_dict["log_post_all"].append(  log_post_std_all)
                if save_dict["PACB_weights"]:
                    self.output_dict["PACB_weights"].append(self.get_model_weights())
                if save_dict["log_prior_std"]:
                    self.output_dict["log_prior_std"].append(log_prior_std)
        return

    def optimize_PACB(self, prior_weights, epochs=20, learning_rate=0.01, batch_size=100, drop_lr=10, lr_factor=0.05,
                      save_dict=None, trainWeights=True):
        """ Optimize the PAC Bayes Bound depending on a prior """

        with self.graph.as_default():
            A, cost, Bquad, effective_m, log_prior_std, log_post_std_list = self.PACB_objective(prior_weights, trainWeights=trainWeights)
            train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
            train_step_dropped = tf.train.RMSPropOptimizer(learning_rate=lr_factor*learning_rate).minimize(cost)

            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            init = tf.initialize_all_variables()
            self.sess.run(init)

            mean_accuracy_stoch_print = []
            mean_accuracy_det_print = []
            self.mean_weights_list = []
            self.var_weights_list = []
            self.pacb_list = []
            Nsamples = self.Nsamples
            trainX, trainY = self.X, self.Y

            for i in range(epochs*int(Nsamples/batch_size)):
                epoch = int(i / (Nsamples/batch_size))
                batch_x, batch_y = next_batch(trainX, trainY, batch_size, int(i % (Nsamples/batch_size)))

                # For the stochastic network
                noise_list = generate_noise(self.layer_shapes)

                if i % int(Nsamples/batch_size) == 0: # At every epoch, shuffle the data
                    trainX, trainY = shuffledata(trainX, trainY)

                    train_accuracy_stoch = 0
                    train_accuracy_det = 0
                    for ib in range(int(Nsamples/batch_size)):
                        bx, by = next_batch(trainX, trainY, batch_size, int(ib % (Nsamples/batch_size)))

                        # Find accuracy of the stochastic network
                        noise_list = generate_noise(self.layer_shapes)
                        feed_input = [self.x, self.y] + self.param_noise_list
                        feed_output = [bx, by] + noise_list
                        feed_dict_val = {i: d for i, d in zip(feed_input, feed_output)}
                        train_accuracy_stoch += self.sess.run(self.accuracySUM, feed_dict=feed_dict_val)

                        # Find accuracy of the deterministic network
                        zero_noise_list = generate_zero_noise(self.layer_shapes)

                        feed_input = [self.x, self.y] + self.param_noise_list
                        feed_output_det = [bx, by] + zero_noise_list
                        feed_dict_val_det = {i: d for i, d in zip(feed_input, feed_output_det)}
                        train_accuracy_det += self.sess.run(self.accuracySUM, feed_dict=feed_dict_val_det)

                    train_accuracy_stoch = train_accuracy_stoch/Nsamples
                    train_accuracy_det = train_accuracy_det/Nsamples
                    mean_accuracy_stoch_print.append(train_accuracy_stoch)
                    mean_accuracy_det_print.append(train_accuracy_det)

                if (drop_lr is not None) and (epoch>drop_lr):
                    train_step = train_step_dropped

                feed_input = [self.x, self.y] + self.param_noise_list
                feed_output = [batch_x, batch_y] + noise_list
                feed_dict_val = {i: d for i, d in zip(feed_input, feed_output)}
                _, A_i, cost_i, kldiv2_i, B_i, m_w, v_w, _log_prior_std, Bquad_i = self.sess.run(
                    [train_step, A, cost, self.KLdivTimes2, self.B, self.mean_weights_component,
                     self.var_weights_component, log_prior_std, Bquad], feed_dict=feed_dict_val)

                # Save at a frequency for every epoch, or at the last run
                if i%(1 * Nsamples / batch_size)==0 or (i == epochs*int(Nsamples/batch_size) - 1):
                    bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)
                    output ="".join("Epoch:" + '%04d' % (epoch+1) + " cost=" + str(cost_i[0]) +
                                    " mean accuracy %.4f" % train_accuracy_stoch + ' KL div:  %.4f' % (kldiv2_i/2) +
                                    ' A term: %.4f' % A_i + ' B term: %.4f' % B_i + ' Bquad: %.4f' % Bquad_i +
                                    ' log_prior_std: %.4f' % _log_prior_std + ' B PAC: %.4f' % bpac)
                    print(output)
                self.PACB_store(save_dict, i=i, log_prior_std=_log_prior_std, m_w=m_w, v_w=v_w, bpac=bpac,
                                log_post_std_list=log_post_std_list, KL_val=kldiv2_i / 2) # Store the desired values for saving

            # Save log_prior_std and log_posterior for evaluate_SNN_accuracy()
            self.log_prior_std = log_prior_std.eval(session=self.sess)
            noisevecs = []
            for log_post_std in log_post_std_list:
              noisevecs.append(log_post_std.eval(session=self.sess))
            self.log_post_all = []
            for nv in noisevecs:
                for v in nv.flatten():
                    self.log_post_all.append(v)

        return

    def evaluate_SNN_accuracy(self, testX, testY, prior_weights=None, N_SNN_samples=200, save_dict=None):
        """ Run the check accuracy code """

        # Hacky way to Create a new graph and session for now, resets the default:
        params_mean_values = self.get_model_weights()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            nparams, parameter_shapes = self.count_N_params()
            effective_m = self.X.shape[0]

            params_means =[]
            for (a,b) in zip(params_mean_values, prior_weights):
              params_means.append(a-b)

            log_post_all = self.log_post_all
            log_post_std_init_list = []
            for (par_shape , par_mean) in zip(self.layer_shapes, params_means):
                log_post_std_init_tmp = np.reshape(log_post_all[:np.prod(par_shape)], par_shape)
                assert log_post_std_init_tmp.shape==par_mean.shape
                log_post_std_init_list.append(log_post_std_init_tmp)
                log_post_all = log_post_all[np.prod(par_shape):]

            # Load and discretize prior variance parameter
            init_log_prior_std = self.log_prior_std
            jdisc = self.log_prior_std_precision * (np.log(self.log_prior_std_base) - 2 * init_log_prior_std)
            print("Before discretization")
            print(init_log_prior_std,jdisc)
            jdisc_up   = np.float32(math.ceil(jdisc))
            jdisc_down = np.float32(math.floor(jdisc))
            init_log_prior_std_up = (np.log(self.log_prior_std_base) - jdisc_up / self.log_prior_std_precision) / 2
            init_log_prior_std_down = (np.log(self.log_prior_std_base) - jdisc_down / self.log_prior_std_precision) / 2
            print("After discretization")
            print(init_log_prior_std_down,jdisc_down)
            print(init_log_prior_std_up,jdisc_up)

            param_noise_list = []
            for par_shape in self.layer_shapes:
                param_noise_list.append(tf.placeholder("float", par_shape))

            self.x, self.y = self.create_placeholders()

            log_prior_std = tf.placeholder("float", ())
            jopt = tf.placeholder("float", ())

            log_post_std_list = []
            for (scopename,n_in,n_out,w_init,b_init) in zip(self.scopes_list,self.layer_shapes[0::2], self.layer_shapes[1::2], log_post_std_init_list[0::2],log_post_std_init_list[1::2]):
                with tf.variable_scope(scopename) as scope:
                    log_post_std_list.append(variable_initializer('log_post_std_w', shape=n_in, initializer=tf.constant_initializer(w_init)))
                    log_post_std_list.append(variable_initializer('log_post_std_b', shape=n_out, initializer=tf.constant_initializer(b_init)))

            norm_post_variance = tf.add_n(list(map(lambda x: tf.reduce_sum(tf.exp(x*2)), log_post_std_list)))
            norm_params = tf.add_n(list(map(lambda x: tf.reduce_sum(x**2), params_means)))
            sum_log_post_variance = tf.add_n(list(map(lambda x: tf.reduce_sum(x), log_post_std_list)))

            network_perturb_list = []
            for (log_post_std,param_noise) in zip(log_post_std_list,param_noise_list):
                network_perturb_list.append(tf.multiply(tf.exp(log_post_std), param_noise))

            yhat, _varout = self.model_with_noise(self.x, network_perturb_list, self.scopes_list, self.layers, params_mean_values, graph=self.graph, trainable=False)

            correct_prediction = tf.equal(tf.cast(yhat >= 0, tf.float32) - tf.cast(yhat < 0, tf.float32),
                                          self.y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            KLdivTimes2 = (norm_post_variance+norm_params)/(tf.exp(2*log_prior_std)) + 2*nparams*log_prior_std - nparams - 2*sum_log_post_variance
            Bquad = KLdivTimes2/2 + tf.log(np.pi**2*effective_m/(6*self.deltaPAC)) \
                                      + 2*tf.log(jopt)
            B = tf.sqrt(Bquad/(2*(effective_m-1)))

            init = tf.initialize_all_variables()

            with self.sess as sess:
                sess.run(init)

                mean_train_accuracy = 0
                mean_test_accuracy = 0
                for ns in range(N_SNN_samples):
                    noise_list = generate_noise(self.layer_shapes)

                    feed_input = [self.x,self.y] + param_noise_list
                    train_accur_i = self.print_accuracy_in_batches_noise(self.X, self.Y, noise_list,
                                                                         feed_input, accuracy, whichset='train')

                    test_accur_i = self.print_accuracy_in_batches_noise(testX, testY, noise_list, feed_input, accuracy,
                                                                        whichset='test')
                    mean_train_accuracy += train_accur_i
                    mean_test_accuracy += test_accur_i

                mean_train_accuracy = mean_train_accuracy/N_SNN_samples
                mean_test_accuracy = mean_test_accuracy/N_SNN_samples
                print("Train error :", (1-mean_train_accuracy),
                      "Test error :", (1-mean_test_accuracy))

                B_valD = B.eval({log_prior_std: init_log_prior_std_down, jopt: jdisc_down})
                B_valU = B.eval({log_prior_std: init_log_prior_std_up, jopt: jdisc_up})
                B_val= np.minimum(B_valU,B_valD)
                if B_valU<B_valD:
                    KL_val = KLdivTimes2.eval({log_prior_std: init_log_prior_std_up, jopt: jdisc_up})/2.0
                else:
                    KL_val = KLdivTimes2.eval({log_prior_std: init_log_prior_std_down, jopt: jdisc_down})/2.0

                bpac = approximate_BPAC_bound(mean_train_accuracy, B_val)
                print("Results with delta = %.2f"%(self.deltaPAC+0.01))
                print("PAC bound error:", '%.4f' % bpac,
                      "Gen bound :", '%.4f' % B_val,
                      "KL value: ", '%.4f' % KL_val)

                self.PACB_store(save_dict, i=0, log_prior_std=self.log_prior_std, bpac=bpac, B_val=B_val, KL_val=KL_val, test_acc=mean_test_accuracy, train_acc=mean_train_accuracy)
        return bpac, B_val, KL_val, self.deltaPAC
