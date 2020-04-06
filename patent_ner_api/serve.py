from model.patent_ner_model import Patent_Ner_Model
from model.config import Config
from model.data_util import load_embeddings,ModelHelper
from model.util import create_conll_str_with_pos,read_conll_4columns_per_patent

def get_model_api():
    """Returns lambda function for api"""

    # 1. initialize model once and for all
    config = Config()
    config.helper = ModelHelper.load(config.tok2id)
    # 2. get pre-trained embeddings
    config.embeddings = load_embeddings(config.filename_words, config.filename_word_vectors, config.helper)
    model  = Patent_Ner_Model(config)
    model.build()
    model.restore_session("results/rnn/model.weights")
    def model_api(input_conll):
        """
        Args:
            input_conll: submitted to the API, raw string

        Returns:
            output_data: after some transformation, to be
                returned to the API


        preds = model.predict(input_data)
        return preds
        """
        input_data, input_data_secton = read_conll_4columns_per_patent(input_conll)

        output = model.output_4columns(input_data)
        sentences, labels, predictions = zip(*output)
        predictions = [[config.LBLS[l] for l in preds] for preds in predictions]
        output = zip(sentences, labels, predictions)
        output_conll = create_conll_str_with_pos(output,input_data_secton)

        #with open('data/result.conll','w') as f:
        #    f.write(output_conll);

        return output_conll;

    return model_api



