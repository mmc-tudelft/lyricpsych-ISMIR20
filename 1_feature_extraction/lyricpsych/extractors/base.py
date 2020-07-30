class BaseTextFeatureExtractor:
    def extract(self, corpus):
        """ Extract text feature for given corpus

        Inputs:
            corpus (lyricpsych.data.Corpus): data object

        Outputs:
            feature (lyricpsych.feature.TextFeature): output
        """
        raise NotImplementedError()


class TextFeature:
    def __init__(self, name, ids, features, columns):
        self.name = name
        self.ids = ids
        self.inv_ids = {
            orig:internal for internal, orig in enumerate(self.ids)
        }
        self.features = features
        self.columns = columns

    def __str__(self):
        return self.name


class TopicFeature(TextFeature):
    def __init__(self, k, ids, doc_topic, topic_term, id2word):
        super().__init__(
            name='TopicFeature@{:d}'.format(k), ids=ids, features=doc_topic,
            columns=['topic{:d}'.format(d+1) for d in range(k)]
        )
        self.topic_terms = topic_term
        self.id2word = id2word
