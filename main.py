
import numpy as np
from random import shuffle

function2grad = {};
function2grad["init"] = init_grad;
function2grad["logsoftmax"] = logsoftmax_grad;
function2grad["weightbias"] = weightbias_grad;
function2grad["emb"] = emb_grad;
function2grad["tanhsigmoideltwisemul"] = tanhsigmoideltwisemul_grad;
function2grad["newc"] = newc_grad;
function2grad["concat"] = concat_grad;


# Load a list of abstract language descriptors
def load_languages(language_file):
    fi = open(language_file, "r")
    lang_list = []

    for line in fi:
        parts = line.strip().split("\t")

        ranking = [int(x) for x in parts[0].split(",")]
        vowel_inventory = parts[1].split(",")
        consonant_inventory = parts[2].split(",")

        lang = [ranking, vowel_inventory, consonant_inventory]

        lang_list.append(lang)

    return lang_list

# Load the file input/output correspondences
def load_io(io_file):
    fi = open(io_file, "r")

    io_correspondences = {}

    for line in fi:
        parts = line.strip().split("\t")
        ranking = tuple([int(x) for x in parts[0].split(",")])

        value = parts[1]
        value_groups = value.split("&")

        value_list = []

        for group in value_groups:
            components = group.split("#")
            inp = components[0]
            outp = components[1]
            steps = components[2].split(",")

            value_list.append([inp, outp, steps])

        io_correspondences[ranking] = value_list

    return io_correspondences
# Load a language that is just Cs and Vs
def load_dataset(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        dev_set = [elt.split(",") for elt in parts[1].split()]
        test_set = [elt.split(",") for elt in parts[2].split()]
        vocab = parts[3].split()
        key_string = parts[4].split(",")

        v_list = key_string[0].split()
        c_list = key_string[1].split()
        ranking = [int(x) for x in key_string[2].split()]

        key = [v_list, c_list, ranking]

        langs.append([train_set, dev_set, test_set, vocab, key])

    return langs

# Load a language that is just Cs and Vs
def load_dataset_scramble(dataset_file):
    fi = open(dataset_file, "r")

    all_train_sets = []
    all_dev_sets = []
    all_test_sets = []

    n_tasks = 0

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        dev_set = [elt.split(",") for elt in parts[1].split()]
        test_set = [elt.split(",") for elt in parts[2].split()]
        all_train_sets += train_set
        all_dev_sets += dev_set
        all_test_sets += test_set

        vocab = parts[3].split()

        n_tasks += 1

    shuffle(all_train_sets)
    shuffle(all_dev_sets)
    shuffle(all_test_sets)

    train_len = len(train_set)
    dev_len = len(dev_set)
    test_len = len(test_set)


    for i in range(n_tasks):
        train_set = all_train_sets[i*train_len:(i+1)*train_len]
        dev_set = all_dev_sets[i*dev_len:(i+1)*dev_len]
        test_set = all_test_sets[i*test_len:(i+1)*test_len]

        v_list = "scrambled"
        c_list = "scrambled"
        ranking = "scrambled"

        key = [v_list, c_list, ranking]

        langs.append([train_set, dev_set, test_set, vocab, key])

    return langs


# Load a language that is just Cs and Vs
def load_dataset_cv(dataset_file):
    fi = open(dataset_file, "r")

    langs = []
    for line in fi:
        parts = line.strip().split("\t")

        train_set = [elt.split(",") for elt in parts[0].split()]
        test_set = [elt.split(",") for elt in parts[1].split()]
        vocab = parts[2].split()

        langs.append([train_set, test_set, vocab])

    return langs


# Break a list into batches of the desired size
def batchify_list(lst, batch_size=100):
    batches = []
    this_batch_in = []
    this_batch_out = []

    for index, elt in enumerate(lst):
        #print(elt)
        this_batch_in.append(elt[0])
        this_batch_out.append(elt[1])

        if (index + 1) % batch_size == 0:
            batches.append([this_batch_in, this_batch_out])
            this_batch_in = []
            this_batch_out = []

    if this_batch_in != []:
        batches.append([this_batch_in, this_batch_out])

    return batches

# Trim the excess from the end of an output string
def process_output(output):
    if "EOS" in output:
        return output[:output.index("EOS")]
    else:
        return output


import random
from random import shuffle
from collections import OrderedDict


# Redefine a basic PyTorch model to allow
# for double gradients and manual modification
# of weights
class ModifiableModule(object):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


    def load_state_dict(self, sdict, same_var=False):
        for name in sdict:
            param = sdict[name]
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)

            self.set_param(name, param)

    def state_dict(self):
        return OrderedDict(self.named_params())

# Redefined linear layer
class GradLinear(ModifiableModule):
    def __init__(self, inp_size, outp_size):
        super(GradLinear, self).__init__()
        self.weights = np.random.rand(outp_size, inp_size)
        self.bias = np.random.rand(outp_size)

    def forward(self, x):

        return np.dot(self.weights,x) + self.bias

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def logsoftmax(x):
    return np.log(softmax(x))

# Redefined LSTM
class GradLSTM(ModifiableModule):
    def __init__(self, input_size, hidden_size):
        super(GradLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wi_bias = np.random.rand(hidden_size)
        self.wf_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wf_bias = np.random.rand(hidden_size)
        self.wg_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wg_bias = np.random.rand(hidden_size)
        self.wo_weights = np.random.rand(hidden_size, hidden_size + input_size)
        self.wo_bias = np.random.rand(hidden_size)


    def forward(self, inp, hidden):
        hx, cx = hidden

        input_plus_hidden = np.concatenate([inp.flatten(), hx.flatten()])

        i_tpre = np.dot(self.wi_weights,input_plus_hidden) + self.wi_bias
        i_t = sigmoid(i_tpre)
        f_tpre = np.dot(self.wf_weights,input_plus_hidden) + self.wf_bias
        f_t = sigmoid(f_tpre)
        g_tpre = np.dot(self.wg_weights,input_plus_hidden) + self.wg_bias
        g_t = tanh(g_tpre)
        o_tpre = np.dot(self.wo_weights,input_plus_hidden) + self.wo_bias
        o_t = sigmoid(o_tpre)

        cx = f_t * cx + i_t * g_t
        hx = o_t * tanh(cx)

        #myhook = input_plus_hidden.register_hook(print_grad)

        return hx, (hx, cx), o_tpre, input_plus_hidden, i_tpre, f_tpre, g_tpre


    def named_leaves(self):
        return [('wi_weights', self.wi_weights), ('wi_bias', self.wi_bias),
                ('wf_weights', self.wf_weights), ('wf_bias', self.wf_bias),
                ('wg_weights', self.wg_weights), ('wg_bias', self.wg_bias),
                ('wo_weights', self.wo_weights), ('wo_bias', self.wo_bias)]

# Redefined embedding layer
class GradEmbedding(ModifiableModule):
    def __init__(self, vocab_size, emb_size):
        super(GradEmbedding, self).__init__()
        self.weights = np.random.rand(emb_size, vocab_size)


    def forward(self, x):
        return np.dot(self.weights,x)

    def named_leaves(self):
        return [('weights', self.weights)]

def onehot(ind):
    oh = np.zeros(34)
    oh[ind] = 1.0

    return oh
# Encoder/decoder model
class EncoderDecoder(ModifiableModule):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = GradEmbedding(vocab_size, input_size)
        self.enc_lstm = GradLSTM(input_size, hidden_size)

        self.dec_lstm = GradLSTM(input_size, hidden_size)
        self.dec_output = GradLinear(hidden_size, vocab_size)

        self.max_length = 20

        self.set_dicts("a e i o u A E I O U b c d f g h j k l m n p q r s t v w x z .".split())


    def forward(self, inp, outp_length=20, corr_outp=None):
        computation_graph = {}
        
        # Initialize the hidden and cell states
        hidden = (np.zeros([1,self.hidden_size]), np.zeros([1,self.hidden_size]))
        
        if corr_outp is not None:
            computation_graph["enc_h-1"] = ["init", [["ZERO", hidden[0]]]]
            computation_graph["enc_c-1"] = ["init", [["ZERO", hidden[1]]]]
        
        cprev_name = "enc_c-1"
        hprev_name = "enc_h-1"

        this_seq = []
        # Iterate over the sequence
        for elt in inp:
            ind = self.char2ind[elt]
            this_seq.append(ind)

        inp_length = len(inp)
        if inp_length > 0:

            # Pass the sequences through the encoder, one character at a time
            for index, elt in enumerate(this_seq):
                cprev_name = "enc_c" + str(index-1)
                hprev_name = "enc_h" + str(index-1)
                
                # Embed the character
                emb = self.embedding.forward(onehot(elt))
                
                if corr_outp is not None:
                    computation_graph["enc_input" + str(index)] = ["emb", [["onehot", elt], ["emb_mat", self.embedding.weights]]]
                    computation_graph["enc_inputhidden" + str(index)] = ["concat", [["enc_input" + str(index), emb], [hprev_name, hidden[0]]]]
                
                # Pass through the LSTM
                output, hidden_new, o_t, iph, i_t, f_t, g_t = self.enc_lstm.forward(emb, hidden)
                hx_new, cx_new = hidden_new
                hidden_prev = hidden
                
                if corr_outp is not None:
                    computation_graph["enc_h" + str(index)] = ["tanhsigmoideltwisemul", [["enc_c" + str(index), cx_new], ["enc_o" + str(index), o_t]]];
                    computation_graph["enc_c" + str(index)] = ["newc", [[cprev_name, hidden_prev[1]], ["enc_f" + str(index), f_t], ["enc_i" + str(index), i_t], ["enc_g" + str(index), g_t]]];
                    computation_graph["enc_o" + str(index)] = ["weightbias", [["enc_inputhidden" + str(index), iph],["enc_wo", self.enc_lstm.wo_weights],["enc_bo", self.enc_lstm.wo_bias]]];
                    computation_graph["enc_f" + str(index)] = ["weightbias", [["enc_inputhidden" + str(index), iph],["enc_wf", self.enc_lstm.wf_weights],["enc_bf", self.enc_lstm.wf_bias]]];
                    computation_graph["enc_i" + str(index)] = ["weightbias", [["enc_inputhidden" + str(index), iph],["enc_wi", self.enc_lstm.wi_weights],["enc_bi", self.enc_lstm.wi_bias]]];
                    computation_graph["enc_g" + str(index)] = ["weightbias", [["enc_inputhidden" + str(index), iph],["enc_wg", self.enc_lstm.wg_weights],["enc_bg", self.enc_lstm.wg_bias]]];

                hidden_prev = hidden
                hidden = hidden_new

        cprev_name = "enc_c" + str(index)
        hprev_name = "enc_h" + str(index)

        encoding = hidden
        # Decoding

        # Previous output characters (used as input for the following time step)
        prev_output = "SOS"

        # Accumulates the output sequences
        out_string = ""



        # Probabilities at each output position (used for computing the loss)
        logits = []
        preds = []
        hiddens = []
        ots = []
        iphs = []
        hidden_prev = hidden
        its = []
        fts = []
        gts = []


        if corr_outp is not None:
            outp_length = len(corr_outp) + 1
            corr_outp = list(corr_outp) + ["EOS"]
        for index in range(min(self.max_length,outp_length)):
            # Determine the previous output character for each element
            # of the batch; to be used as the input for this time step

            # Embed the previous outputs
            emb = self.embedding.forward(onehot(self.char2ind[prev_output]))
            
            if corr_outp is not None:
                computation_graph["dec_input" + str(index)] = ["emb", [["onehot", self.char2ind[prev_output]], ["emb_mat", self.embedding.weights]]];
                computation_graph["dec_inputhidden" + str(index)] = ["concat", [["dec_input" + str(index), emb], [hprev_name, hidden[0]]]]


            # Pass through the decoder
            output, hidden, o_t, iph, i_t, f_t, g_t = self.dec_lstm.forward(emb, hidden)
            hx_new, cx_new = hidden
            
            #myhook = o_t.register_hook(print_grad)

            # Determine the output probabilities used to make predictions
            pred = self.dec_output.forward(output.flatten())
            probs = logsoftmax(pred)
            
            if corr_outp is not None:
                print(corr_outp, corr_outp[index])
                computation_graph["logit" + str(index)] = ["logsoftmax", [["pred" + str(index), pred], self.char2ind[corr_outp[index]]]];
                computation_graph["pred" + str(index)] = ["weightbias", [["dec_h" + str(index), hx_new],["output_weights", self.dec_output.weights],["output_bias", self.dec_output.bias]]];

                computation_graph["dec_h" + str(index)] = ["tanhsigmoideltwisemul", [["dec_c" + str(index), cx_new], ["dec_o" + str(index), o_t]]];
                computation_graph["dec_c" + str(index)] = ["newc", [[cprev_name, hidden_prev[1]], ["dec_f" + str(index), f_t], ["dec_i" + str(index), i_t], ["dec_g" + str(index), g_t]]];
                computation_graph["dec_o" + str(index)] = ["weightbias", [["dec_inputhidden" + str(index), iph],["dec_wo", self.dec_lstm.wo_weights],["dec_bo", self.dec_lstm.wo_bias]]];
                computation_graph["dec_f" + str(index)] = ["weightbias", [["dec_inputhidden" + str(index), iph],["dec_wf", self.dec_lstm.wf_weights],["dec_bf", self.dec_lstm.wf_bias]]];
                computation_graph["dec_i" + str(index)] = ["weightbias", [["dec_inputhidden" + str(index), iph],["dec_wi", self.dec_lstm.wi_weights],["dec_bi", self.dec_lstm.wi_bias]]];
                computation_graph["dec_g" + str(index)] = ["weightbias", [["dec_inputhidden" + str(index), iph],["dec_wg", self.dec_lstm.wg_weights],["dec_bg", self.dec_lstm.wg_bias]]];
            
            print(cprev_name, hprev_name)
            cprev_name = "dec_c" + str(index);
            hprev_name = "dec_h" + str(index);
            print(cprev_name, hprev_name)
            print("")
            
            logits.append(probs)
            preds.append(pred)
            hiddens.append(hidden)
            ots.append(o_t)
            iphs.append(iph)
            its.append(i_t)
            fts.append(f_t)
            gts.append(g_t)

            # Discretize the output labels (via argmax) for generating an output character
            label = np.argmax(probs)

            char = self.ind2char[label]
            out_string += char
            prev_output = char
            hidden_prev = hidden


        return out_string, logits, encoding, preds, hiddens, ots, iphs, hidden_prev, its, fts, gts, computation_graph

    def named_submodules(self):
        return [('embedding', self.embedding), ('enc_lstm', self.enc_lstm),
                ('dec_lstm', self.dec_lstm), ('dec_output', self.dec_output)]

    # Create a copy of the model
    def create_copy(self, same_var=False):
        new_model = EncoderDecoder(self.vocab_size, self.input_size, self.hidden_size)
        new_model.copy(self, same_var=same_var)

        return new_model

    def set_dicts(self, vocab_list):
        vocab_list = ["NULL", "SOS", "EOS"] + vocab_list

        index = 0
        char2ind = {}
        ind2char = {}

        for elt in vocab_list:
            char2ind[elt] = index
            ind2char[index] = elt
            index += 1

        self.char2ind = char2ind
        self.ind2char = ind2char

def set_params():
    encdec = EncoderDecoder(34,10,256)

    encdec.enc_lstm.wo_weights = np.loadtxt("enc_lstm.wo_weights")
    encdec.enc_lstm.wi_weights = np.loadtxt("enc_lstm.wi_weights")
    encdec.enc_lstm.wg_weights = np.loadtxt("enc_lstm.wg_weights")
    encdec.enc_lstm.wf_weights = np.loadtxt("enc_lstm.wf_weights")
    encdec.enc_lstm.wo_bias = np.loadtxt("enc_lstm.wo_bias")
    encdec.enc_lstm.wi_bias = np.loadtxt("enc_lstm.wi_bias")
    encdec.enc_lstm.wg_bias = np.loadtxt("enc_lstm.wg_bias")
    encdec.enc_lstm.wf_bias = np.loadtxt("enc_lstm.wf_bias")

    encdec.dec_lstm.wo_weights = np.loadtxt("dec_lstm.wo_weights")
    encdec.dec_lstm.wi_weights = np.loadtxt("dec_lstm.wi_weights")
    encdec.dec_lstm.wg_weights = np.loadtxt("dec_lstm.wg_weights")
    encdec.dec_lstm.wf_weights = np.loadtxt("dec_lstm.wf_weights")
    encdec.dec_lstm.wo_bias = np.loadtxt("dec_lstm.wo_bias")
    encdec.dec_lstm.wi_bias = np.loadtxt("dec_lstm.wi_bias")
    encdec.dec_lstm.wg_bias = np.loadtxt("dec_lstm.wg_bias")
    encdec.dec_lstm.wf_bias = np.loadtxt("dec_lstm.wf_bias")

    encdec.embedding.weights = np.loadtxt("embedding.weights").transpose()
    encdec.dec_output.weights = np.loadtxt("dec_output.weights")
    encdec.dec_output.bias = np.loadtxt("dec_output.bias")


    
def init_grad(args, result):
    return []

def logsoftmax_grad(args):
    name = args[1][0][0]
    pred = args[1][0][1]
    correct_ind = args[1][1]

    print(name)
    print(pred)

    onehot_vec = onehot(correct_ind)

    sm = softmax(pred);

    sm_expand = np.zeros([34,34])
    for k in range(34):
        sm_expand[k] = sm

    sm_expand = sm_expand.transpose()
    print(sm_expand)

    mat_grad = sm_expand - np.identity(34)

    grad_pred = np.dot(onehot_vec, mat_grad.transpose());

    return [[name, grad_pred]];


def weightbias_grad(args, result):
    name_inp = args[0][0]
    inp = unsqueeze(args[0][1])

    name_weight = args[1][0]
    weight = args[1][1]

    name_bias = args[2][0]
    bias = args[2][1]


    grad_bias = result

    grad_weight = np.dot(inp.transpose(), unsqueeze(grad_bias)).transpose()
    grad_inp = np.dot(grad_bias, weight)

    return [[name_weight, grad_weight], [name_bias, grad_bias], [name_inp, grad_inp]]

def emb_grad(args, result):
    #print(args)

    name_ind = args[0][0]
    ind = args[0][1]

    name_weight = args[1][0]
    weight = args[1][1]

    onehot_vec = unsqueeze(onehot(ind))

    grad_weight = np.dot(onehot_vec.transpose(),result)

    return [[name_weight, grad_weight]]

def unsqueeze(vec):
    return np.reshape(vec, (1,-1))

def tanhsigmoideltwisemul_grad(args, result):

    name_c = args[0][0];
    ct = unsqueeze(args[0][1]);

    name_o = args[1][0];
    ot = unsqueeze(args[1][1]);

    grad_ot = sigmoid(ot) * (np.ones(ot.size) - sigmoid(ot)) * tanh(ct) * result
    grad_ct = sigmoid(ot) * result * (np.ones(ct.size) - np.power(tanh(ct), 2))


    return [[name_c, grad_ct], [name_o, grad_ot]];

def newc_grad(args, result):

    name_cprev = args[0][0]
    cprev = unsqueeze(args[0][1])

    name_f = args[1][0]
    ft = unsqueeze(args[1][1])

    name_i = args[2][0]
    it = unsqueeze(args[2][1])

    name_g = args[3][0]
    gt = unsqueeze(args[3][1])


    grad_cprev = result * sigmoid(ft);
    grad_ft = cprev * result * sigmoid(ft) * (np.ones(ft.size) - sigmoid(ft))
    grad_it = sigmoid(it) * (np.ones(it.size) - sigmoid(it)) *  tanh(gt) * result
    grad_gt = (np.ones(gt.size) - np.power(tanh(gt), 2)) * sigmoid(it) * result

    return [[name_cprev, grad_cprev], [name_f, grad_ft], [name_i, grad_it], [name_g, grad_gt]];

def concat_grad(args, result):

    inp_name = args[0][0]
    inp = unsqueeze(args[0][1])

    hprev_name = args[1][0]
    hprev = unsqueeze(args[1][1])


    grad_inp = unsqueeze(result[0][:10])

    grad_hprev = unsqueeze(result[0][10:])

    return [[inp_name, grad_inp], [hprev_name, grad_hprev]];



def init_grads():
    gradients = {}
    gradients["dec_wi"] = None
    gradients["dec_wf"] = None
    gradients["dec_wg"] = None
    gradients["dec_wo"] = None

    gradients["dec_bi"] = None
    gradients["dec_bf"] = None
    gradients["dec_bg"] = None
    gradients["dec_bo"] = None

    gradients["enc_wi"] = None
    gradients["enc_wf"] = None
    gradients["enc_wg"] = None
    gradients["enc_wo"] = None

    gradients["enc_bi"] = None
    gradients["enc_bf"] = None
    gradients["enc_bg"] = None
    gradients["enc_bo"] = None

    gradients["output_weights"] = None
    gradients["output_bias"] = None

    gradients["emb_mat"] = None

    return gradients


def update_params(model, lr):
    model.enc_lstm.wo_weights = model.enc_lstm.wo_weights - lr * gradients["enc_wo"]
    model.enc_lstm.wi_weights = model.enc_lstm.wi_weights - lr * gradients["enc_wi"] 
    model.enc_lstm.wg_weights = model.enc_lstm.wg_weights - lr * gradients["enc_wg"]
    model.enc_lstm.wf_weights = model.enc_lstm.wf_weights - lr * gradients["enc_wf"]
    model.enc_lstm.wo_bias = model.enc_lstm.wo_bias - lr * gradients["enc_bo"]
    model.enc_lstm.wi_bias = model.enc_lstm.wi_bias - lr * gradients["enc_bi"]
    model.enc_lstm.wg_bias = model.enc_lstm.wg_bias - lr * gradients["enc_bg"]
    model.enc_lstm.wf_bias = model.enc_lstm.wf_bias - lr * gradients["enc_bf"]

    model.dec_lstm.wo_weights = model.dec_lstm.wo_weights - lr * gradients["dec_wo"]
    model.dec_lstm.wi_weights = model.dec_lstm.wi_weights - lr * gradients["dec_wi"] 
    model.dec_lstm.wg_weights = model.dec_lstm.wg_weights - lr * gradients["dec_wg"]
    model.dec_lstm.wf_weights = model.dec_lstm.wf_weights - lr * gradients["dec_wf"]
    model.dec_lstm.wo_bias = model.dec_lstm.wo_bias - lr * gradients["dec_bo"]
    model.dec_lstm.wi_bias = model.dec_lstm.wi_bias - lr * gradients["dec_bi"]
    model.dec_lstm.wg_bias = model.dec_lstm.wg_bias - lr * gradients["dec_bg"]
    model.dec_lstm.wf_bias = model.dec_lstm.wf_bias - lr * gradients["dec_bf"]

    model.embedding.weights = model.embedding.weights - lr * gradients["emb_mat"].transpose()
    model.dec_output.weights = model.dec_output.weights - lr * gradients["output_weights"]
    model.dec_output.bias = model.dec_output.bias - lr * gradients["output_bias"]

    gradients = init_grads()

import copy

def backprop_gradient(cg, names, gradients):
    done = False

    results = {}

    
    for nameouter in names:
        grad_type = cg[nameouter][0]
        args = cg[nameouter][1]
        grad_function = function2grad[grad_type]

        to_add = grad_function(cg[nameouter])

        for result in to_add:
            name = result[0]
            grad = result[1]

            if name in results:
                results[name] = results[name] + grad
            else:
                results[name] = grad

    while not done:

        if len(results.keys()) == 0:
            done = True
            break


        results_new = {}
        this_len = len(results.keys())
        this_result_list = results.keys()
        print(this_result_list)
        #print(results)

        for name in results.keys():
            grad = results[name];

            if name in gradients:


                if gradients[name] is None:
                    gradients[name] = grad;
                else:
                    gradients[name] = gradients[name] + grad
                    print(name)


            else:
                #print(name)
                grad_type = cg[name][0]
                args = cg[name][1]
                grad_function = function2grad[grad_type]
                to_add = grad_function(args, grad);

                for result in to_add:
                    name = result[0]
                    grad = result[1]


                    if name in results_new:
                        results_new[name] = results_new[name] + grad
                    else:
                        results_new[name] = grad




        results = copy.deepcopy(results_new)



encdec = EncoderDecoder(34,10,256)

encdec.enc_lstm.wo_weights = np.loadtxt("enc_lstm.wo_weights")
encdec.enc_lstm.wi_weights = np.loadtxt("enc_lstm.wi_weights")
encdec.enc_lstm.wg_weights = np.loadtxt("enc_lstm.wg_weights")
encdec.enc_lstm.wf_weights = np.loadtxt("enc_lstm.wf_weights")
encdec.enc_lstm.wo_bias = np.loadtxt("enc_lstm.wo_bias")
encdec.enc_lstm.wi_bias = np.loadtxt("enc_lstm.wi_bias")
encdec.enc_lstm.wg_bias = np.loadtxt("enc_lstm.wg_bias")
encdec.enc_lstm.wf_bias = np.loadtxt("enc_lstm.wf_bias")

encdec.dec_lstm.wo_weights = np.loadtxt("dec_lstm.wo_weights")
encdec.dec_lstm.wi_weights = np.loadtxt("dec_lstm.wi_weights")
encdec.dec_lstm.wg_weights = np.loadtxt("dec_lstm.wg_weights")
encdec.dec_lstm.wf_weights = np.loadtxt("dec_lstm.wf_weights")
encdec.dec_lstm.wo_bias = np.loadtxt("dec_lstm.wo_bias")
encdec.dec_lstm.wi_bias = np.loadtxt("dec_lstm.wi_bias")
encdec.dec_lstm.wg_bias = np.loadtxt("dec_lstm.wg_bias")
encdec.dec_lstm.wf_bias = np.loadtxt("dec_lstm.wf_bias")

encdec.embedding.weights = np.loadtxt("embedding.weights").transpose()
encdec.dec_output.weights = np.loadtxt("dec_output.weights")
encdec.dec_output.bias = np.loadtxt("dec_output.bias")


gradients = init_grads()

