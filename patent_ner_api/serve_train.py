from model.patent_ner_model import Patent_Ner_Model
from model.config import Config
from model.data_util import load_and_preprocess_data_4columns,load_embeddings,ModelHelper
import logging

def do_train():
    # Set up some parameters.
    config = Config()
    config.helper = ModelHelper.load(config.tok2id)
    # 2. get pre-trained embeddings
    config.embeddings = load_embeddings(config.filename_words, config.filename_word_vectors, config.helper)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data_4columns(config.data_train,config.data_dev)
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    model = Patent_Ner_Model(config)
    model.build()
    model.fit(train,dev)


def get_model_api():
    """Returns lambda function for api"""

    # 1. initialize model once and for all
    config = Config()
    model  = Patent_Ner_Model(config)
    model.build()

    model.restore_session("results/rnn/model.weights/")


    def model_api(input_data):
        """
        Args:
            input_data: submitted to the API, raw string

        Returns:
            output_data: after some transformation, to be
                returned to the API

        """
        preds = model.predict(input_data)
        return preds

    return model_api


if __name__=='__main__':
    do_train();