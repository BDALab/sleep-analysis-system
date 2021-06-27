from pandas import DataFrame


class HilevNormLists:

    def __init__(self):
        self.IDs = []
        self.SOLs = []
        self.WKS5 = []
        self.WASOs = []
        self.SEs = []

    def clear(self):
        self.IDs.clear()
        self.SOLs.clear()
        self.WKS5.clear()
        self.WASOs.clear()
        self.SEs.clear()

    def to_data_frame(self):
        cols = ['ID', 'SOL', 'WKS5', 'WASO', 'SE']
        data = {
            'ID': self.IDs,
            'SOL': self.SOLs,
            'WKS5': self.WKS5,
            'WASO': self.WASOs,
            'SE': self.SEs
        }
        return DataFrame(data, columns=cols)
