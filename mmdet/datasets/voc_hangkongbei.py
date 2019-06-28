from .xml_style import XMLDataset

class voc_hangkongbeiDataset(XMLDataset):
    CLASSES = ('Vehicle',)
    def __init__(self, **kwargs):
        super(voc_hangkongbeiDataset, self).__init__(**kwargs)