from pandas import DataFrame


class HilevLists:

    def __init__(self):
        self.IDs = []
        self.TSTs = []
        self.WASOs = []
        self.SEs = []
        self.SFs = []
        self.SOLs = []
        self.WKS5 = []
        self.DTSTs = []

    def clear(self):
        self.IDs.clear()
        self.TSTs.clear()
        self.WASOs.clear()
        self.SEs.clear()
        self.SFs.clear()
        self.SOLs.clear()
        self.WKS5.clear()
        self.DTSTs.clear()

    def to_data_frame(self):
        cols = ['ID', 'TST', 'WASO', 'SE', 'SF', 'SOL', 'WKS5', 'DTST']
        data = {
            'ID': self.IDs,
            'TST': self.TSTs,
            'WASO': self.WASOs,
            'SE': self.SEs,
            'SF': self.SFs,
            'SOL': self.SOLs,
            'WKS5': self.WKS5,
            'DTST': self.DTSTs
        }
        return DataFrame(data, columns=cols)
