from haystack.reader.farm import FARMReader

def get_reader():
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    return reader
