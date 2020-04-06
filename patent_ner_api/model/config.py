import os
from .util import get_logger
from .util import one_hot,load_vocab

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.log_output)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)



    # general config
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    max_length = 300 # longest sequence to parse
    n_classes = 35
    dropout = 0.5
    embed_size = 100
    hidden_size = 300
    batch_size = 32
    n_epochs = 20
    max_grad_norm = 10.
    lr = 0.001
    cell = "rnn"
    output_path = "results/rnn/"

    model_output = output_path + "model.weights"
    eval_output = output_path + "results.txt"
    log_output = output_path + "log"
    filename_tags = "data/tags.txt"
    LBLS = []
    with open(filename_tags,"r") as f:
        lines = f.readlines()
        for line in lines:
            LBLS.append(line.strip())
    filename_words = "data/vocab_patentEmbedding.txt"
    filename_word_vectors = "data/wordVectors_patentEmbedding.txt"
    tok2id = "data"
    NONE = "O"
    LMAP = {k: one_hot(35, i) for i, k in enumerate(LBLS)}
    NUM = "NNNUMMM"
    UNK = "UUUNKKK"

    EMBED_SIZE = 100

    data_train = "data/fig_id_train.conll"
    data_dev = "data/fig_id_validation.conll"


